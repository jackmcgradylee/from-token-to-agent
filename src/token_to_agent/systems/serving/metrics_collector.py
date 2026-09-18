"""Per-request metrics + aggregate latency stats for a serving run.

Independent of the engine so the load generator can mock the same
contract: every request gets a single `RequestRecord` once it
completes (or fails), and the collector rolls up p50/p95/max
distributions for TTFT, TPOT, end-to-end.

All times are in **seconds** (float). Convert to ms at the reporting
boundary, never inside the collector.

Why no numpy / pandas: this runs in tight loops in the CPU simulator
and we don't want to depend on torch + pandas + numpy just to compute
percentiles. The bottleneck for any real vLLM deployment is the GPU,
not the metrics path.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class RequestRecord:
    """A single request's lifecycle, completed.

    `arrival_s` is wall-clock when the request entered the queue.
    `first_token_s` is when the engine emitted the first decoded token.
    `completion_s` is when the engine emitted the final token (or
    the request was preempted / failed).
    """

    request_id: int
    prompt_len: int
    output_len: int            # tokens actually generated (0 if preempted)
    arrival_s: float
    first_token_s: float
    completion_s: float
    preempted: bool = False
    error: str | None = None

    @property
    def ttft_s(self) -> float:
        return max(self.first_token_s - self.arrival_s, 0.0)

    @property
    def e2e_s(self) -> float:
        return max(self.completion_s - self.arrival_s, 0.0)

    @property
    def tpot_s(self) -> float:
        if self.output_len <= 1:
            return 0.0  # not enough decode steps to compute per-token time
        decode_time = max(self.completion_s - self.first_token_s, 0.0)
        return decode_time / (self.output_len - 1)


@dataclass
class AggregateStats:
    """p50 / p95 / max / mean / count for one metric over a set of requests.

    `label` is the metric name (e.g. "TTFT_ms"); `unit` is "ms" or
    "tokens/s". We compute percentiles by linear interpolation on
    the sorted samples; this is the standard "type 7" quantile used
    by numpy / R.
    """

    label: str
    unit: str
    samples: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    def _quantile(self, q: float) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        if len(s) == 1:
            return s[0]
        pos = q * (len(s) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return s[lo]
        frac = pos - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    @property
    def p50(self) -> float:
        return self._quantile(0.50)

    @property
    def p95(self) -> float:
        return self._quantile(0.95)

    @property
    def p99(self) -> float:
        return self._quantile(0.99)

    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def min(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples) if self.samples else 0.0

    def add(self, value: float) -> None:
        self.samples.append(float(value))

    def as_row(self) -> dict[str, float]:
        return {
            f"{self.label}_p50_{self.unit}": self.p50,
            f"{self.label}_p95_{self.unit}": self.p95,
            f"{self.label}_p99_{self.unit}": self.p99,
            f"{self.label}_max_{self.unit}": self.max,
            f"{self.label}_mean_{self.unit}": self.mean,
            f"{self.label}_count": self.count,
        }


class MetricsCollector:
    """Accumulates RequestRecords and computes aggregate stats.

    The simulator calls `record_finished(req)` once per completed
    request; the experiment runner calls `summarise()` to dump
    a JSON-friendly dict.
    """

    def __init__(self) -> None:
        self.records: list[RequestRecord] = []
        # Aggregate buckets built incrementally so we don't re-sort
        # in tight loops. Built on demand by `summarise`.
        self._ttft_ms: AggregateStats | None = None
        self._tpot_ms: AggregateStats | None = None
        self._e2e_ms: AggregateStats | None = None

    def record_finished(self, record: RequestRecord) -> None:
        self.records.append(record)

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.records if not r.preempted and r.error is None)

    @property
    def n_preempted(self) -> int:
        return sum(1 for r in self.records if r.preempted)

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.records if r.error is not None)

    @property
    def n_total_input_tokens(self) -> int:
        return sum(r.prompt_len for r in self.records)

    @property
    def n_total_output_tokens(self) -> int:
        return sum(r.output_len for r in self.records)

    def _ensure_buckets(self) -> None:
        if self._ttft_ms is not None:
            return
        self._ttft_ms = AggregateStats(label="TTFT", unit="ms")
        self._tpot_ms = AggregateStats(label="TPOT", unit="ms")
        self._e2e_ms = AggregateStats(label="E2E", unit="ms")
        for r in self.records:
            if r.preempted or r.error:
                continue
            self._ttft_ms.add(r.ttft_s * 1000.0)
            self._tpot_ms.add(r.tpot_s * 1000.0)
            self._e2e_ms.add(r.e2e_s * 1000.0)

    def summarise(
        self,
        wallclock_s: float,
        gpu_memory_peak_gb: float | None = None,
        cost_per_hour_usd: float | None = None,
    ) -> dict:
        """One-stop report dict: per-metric stats + throughput + cost.

        `wallclock_s` is the duration of the simulated run.
        """
        self._ensure_buckets()
        assert self._ttft_ms is not None and self._tpot_ms is not None and self._e2e_ms is not None

        out: dict = {}
        out.update(self._ttft_ms.as_row())
        out.update(self._tpot_ms.as_row())
        out.update(self._e2e_ms.as_row())

        out["n_total_requests"] = len(self.records)
        out["n_completed"] = self.n_completed
        out["n_preempted"] = self.n_preempted
        out["n_errors"] = self.n_errors
        out["total_input_tokens"] = self.n_total_input_tokens
        out["total_output_tokens"] = self.n_total_output_tokens

        # Throughput: tokens/s over the wallclock.
        if wallclock_s > 0:
            out["throughput_input_tps"] = self.n_total_input_tokens / wallclock_s
            out["throughput_output_tps"] = self.n_total_output_tokens / wallclock_s
            out["throughput_requests_per_s"] = len(self.records) / wallclock_s
        else:
            out["throughput_input_tps"] = 0.0
            out["throughput_output_tps"] = 0.0
            out["throughput_requests_per_s"] = 0.0

        if gpu_memory_peak_gb is not None:
            out["gpu_memory_peak_gb"] = gpu_memory_peak_gb

        if cost_per_hour_usd is not None and wallclock_s > 0:
            cost = cost_per_hour_usd * wallclock_s / 3600.0
            out["cost_usd_total"] = cost
            if self.n_total_input_tokens > 0:
                out["cost_usd_per_1m_input"] = cost * 1e6 / self.n_total_input_tokens
            else:
                out["cost_usd_per_1m_input"] = 0.0
            if self.n_total_output_tokens > 0:
                out["cost_usd_per_1m_output"] = cost * 1e6 / self.n_total_output_tokens
            else:
                out["cost_usd_per_1m_output"] = 0.0
            if self.n_completed > 0:
                out["cost_usd_per_successful_task"] = cost / self.n_completed
            else:
                out["cost_usd_per_successful_task"] = 0.0

        return out


def write_summary_csv(summaries: Iterable[dict], columns: list[str] | None = None) -> str:
    """Format a list of summary dicts as a CSV string with stable columns."""
    import csv
    import io

    rows = list(summaries)
    if not rows:
        return ""
    if columns is None:
        # Stable order: insertion order of the first row, then append any
        # columns that appear only in later rows.
        seen: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        columns = seen

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def write_summary_md(summaries: Iterable[dict], groupby: str | None = None) -> str:
    """Format a list of summary dicts as a Markdown table."""
    rows = list(summaries)
    if not rows:
        return "_no data_\n"
    columns = list(rows[0].keys())
    out = ["| " + " | ".join(columns) + " |"]
    out.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"
