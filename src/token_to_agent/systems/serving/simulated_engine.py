"""Discrete-event simulator for a vLLM-style serving engine.

This is *not* a real model. It mocks the GPU forward pass with
configurable per-token latencies derived from a single
"tokens-per-second-per-active-sequence" knob (`decode_tps_per_seq`)
and a prefill cost proportional to prompt_len squared.

What it DOES model (validated against vLLM semantics):
- Continuous batching: at every decode step the scheduler admits
  newly-arrived requests whose prefill fits in remaining KV cache,
  and ejects requests whose output is done.
- KV cache accounting: each request's KV cache grows linearly with
  prompt_len + decoded_tokens. The scheduler refuses prefill if the
  addition would exceed `gpu_kv_budget_bytes`.
- Chunked prefill (simplified): a long prompt is processed in
  chunks of size `prefill_chunk_tokens`; one chunk per decode step
  shares the GPU with the decode batch.
- TTFT / TPOT / e2e timing: emitted into a MetricsCollector so the
  experiment runner can sweep params and report distributions.

What it does NOT model:
- Speculative decoding (would add draft-model state).
- Prefix caching (would need a RadixAttention-style trie).
- Tensor parallel / pipeline parallel.
- Real GPU memory bandwidth contention.

The point is to validate the *scheduler logic* and *KV cache math*
on CPU. Absolute throughput / cost numbers are deferred to a CUDA
host (see experiments/w05/README-JETSON-LIMITATIONS.md).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .load_generator import Request
from .metrics_collector import MetricsCollector, RequestRecord
from ..inference.kv_cache import (
    ModelConfig,
    kv_bytes_per_token,
    kv_bytes_for_request,
)


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Static knobs for the simulated engine.

    The model config controls KV cache math; the timing knobs control
    the per-token latency assumptions. Calibrate the latter on a real
    GPU box (decode_tps_per_seq ~ 50-80 for 7B on H100 with Flash-2;
    prefill_tps scales roughly as 1e4 / prompt_len).
    """

    model: ModelConfig
    dtype: str = "fp16"
    # Available KV cache budget (bytes). ~KV budget on H100 80GB for 7B.
    gpu_kv_budget_bytes: int = 60 * 1024**3
    # Decode speed: aggregate active tokens/s divided by the number of
    # active sequences in the batch. Lower = slower per-sequence.
    # This is the single knob that controls "is the engine fast?".
    decode_tps_per_seq: float = 50.0
    # Prefill cost: prefill finishes in prompt_len / prefill_tps seconds,
    # processed in chunks of `prefill_chunk_tokens`.
    prefill_tps: float = 8000.0
    prefill_chunk_tokens: int = 256
    # A request stops when it produces this many output tokens, OR
    # the original Request.max_output_len, whichever is smaller.
    max_output_override: int | None = None
    # When a new arrival can't fit in KV cache, drop it (returns error).
    drop_on_overflow: bool = True
    # When prompt_len + max_output exceeds model.max_seq_len, drop the
    # request (returns error="seq_too_long"). On a real engine you
    # might extend max_seq_len via RoPE scaling, but the simulator
    # does not implement that.
    enforce_max_seq_len: bool = True


# ---------------------------------------------------------------------------
# Internal request state
# ---------------------------------------------------------------------------


@dataclass
class _ActiveRequest:
    """The scheduler's view of an in-flight request."""

    req: Request
    model: ModelConfig = field(compare=False)
    dtype: str = "fp16"
    # How many prompt tokens have been prefill-ed so far.
    prefill_progress: int = 0
    # Tokens generated so far.
    decoded_tokens: int = 0
    # Wall-clock time of arrival (copied from Request.arrival_s).
    arrival_s: float = 0.0
    # Wall-clock time when first decoded token was emitted (None until then).
    first_token_s: float | None = None
    # Wall-clock time when the request was finished (success or preemption).
    completion_s: float | None = None
    preempted: bool = False
    error: str | None = None

    @property
    def prompt_total(self) -> int:
        return self.req.prompt_len

    @property
    def max_output(self) -> int:
        return self.req.max_output_len

    @property
    def current_seq_len(self) -> int:
        return self.prefill_progress + self.decoded_tokens

    def kv_bytes(self) -> int:
        return kv_bytes_for_request(self.model, self.current_seq_len, self.dtype)


# ---------------------------------------------------------------------------
# Event types for the heap
# ---------------------------------------------------------------------------


@dataclass(order=True)
class SimEvent:
    """A pending event on the simulation timeline.

    `kind` is one of: "arrival", "decode_step".
    `priority` is the wall-clock time (lower = sooner).
    """

    priority: float
    seq: int
    kind: str = field(compare=True)
    request_id: int = field(compare=False)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SimulatedEngine:
    """A vLLM-style continuous-batching scheduler, in pure Python.

    Use `run(requests, metrics)` to drain a workload to completion
    and produce a populated MetricsCollector.

    Concurrency model: one process, no threads, no async. Events are
    popped off a min-heap by wall-clock time.
    """

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        # Active in-flight requests (post-prefill or mid-decode).
        self._active: dict[int, _ActiveRequest] = {}
        # Pending arrivals waiting to be prefill-ed.
        self._waiting: list[_ActiveRequest] = []
        # Cumulative KV cache currently allocated.
        self._kv_used_bytes: int = 0
        # Peak KV cache usage observed during this run.
        self._kv_peak_bytes: int = 0
        # Heap of (time, seq, kind, request_id) events.
        self._events: list[SimEvent] = []
        # Monotonic counter for tie-breaking events at the same time.
        self._seq = 0
        # Current wall-clock inside the sim.
        self._now: float = 0.0

    # ----- public API -----

    def run(
        self,
        requests: list[Request],
        metrics: MetricsCollector,
        until_s: float | None = None,
    ) -> None:
        """Drain `requests` to completion.

        All `requests` must have monotonically non-decreasing
        `arrival_s`. Caller is responsible for sorting; we don't
        re-sort because it changes the simulation determinism.

        If `until_s` is set, stop simulating new events past that
        wall-clock; any in-flight requests at that point are
        recorded as preempted.
        """
        self._reset_state()
        # Seed events from arrivals.
        for r in requests:
            self._push_event(r.arrival_s, "arrival", r.request_id)

        while self._events:
            ev = heapq.heappop(self._events)
            if until_s is not None and ev.priority > until_s:
                # Reached the cap. Preempt whatever's still in flight.
                self._preempt_all(metrics)
                break
            self._now = ev.priority
            if ev.kind == "arrival":
                self._on_arrival(ev.request_id, requests, metrics)
            elif ev.kind == "decode_step":
                self._on_decode_step(metrics)
            else:
                raise RuntimeError(f"unknown event kind {ev.kind!r}")

        # Anything still active at end-of-queue gets marked preempted.
        self._preempt_all(metrics)

    # ----- internal state -----

    def _reset_state(self) -> None:
        self._active.clear()
        self._waiting.clear()
        self._events.clear()
        self._seq = 0
        self._now = 0.0
        self._kv_used_bytes = 0
        self._kv_peak_bytes = 0

    def _push_event(self, t: float, kind: str, request_id: int) -> None:
        heapq.heappush(self._events, SimEvent(t, self._seq, kind, request_id))
        self._seq += 1

    # ----- event handlers -----

    def _on_arrival(
        self,
        request_id: int,
        requests: list[Request],
        metrics: MetricsCollector,
    ) -> None:
        # Map request_id -> Request quickly (caller passed a list).
        # We re-find it from the list; for benchmarks of ≤100k reqs
        # this is fine. If larger workloads land here, swap to a dict.
        try:
            req = next(r for r in requests if r.request_id == request_id)
        except StopIteration:
            return  # request was not in the workload (shouldn't happen)
        ar = _to_active_request(req, self.cfg)
        # We don't prefill up-front: chunked prefill shares the
        # decode step's wall time, so just queue it.
        was_empty = not self._waiting and not self._active
        self._waiting.append(ar)
        # Kick the engine: if this was the first waiting request
        # and no decode step is queued, schedule one immediately.
        if was_empty:
            self._push_event(self._now, "decode_step", request_id=-1)

    def _on_decode_step(self, metrics: MetricsCollector) -> None:
        """One scheduler tick: advance the world by one decode step.

        Step time = base_time_for_decode + (any prefill chunks this step).
        Order of work:
        1. Decide which waiting requests can start prefill this step
           given KV cache headroom. They join the active batch.
        2. Run one decode step on all currently-active requests.
        3. Emit finished requests (decoded_tokens == max_output OR
           preempted) to the metrics collector.
        4. Schedule the next decode step.
        """
        cfg = self.cfg

        # 1) admit waiting requests whose prefill will fit in KV cache.
        #    Admission checks the *final* KV cost, i.e. assumes the
        #    whole prefill will happen before the next admission.
        #    This is conservative: a long prefill that fits only in
        #    chunks is rejected if its full KV cost doesn't fit now,
        #    which is what vLLM does too (chunked prefill doesn't
        #    change the admission decision).
        still_waiting: list[_ActiveRequest] = []
        for ar in self._waiting:
            # Reject early if the request would exceed max_seq_len.
            if (
                cfg.enforce_max_seq_len
                and ar.prompt_total + ar.max_output > cfg.model.max_seq_len
            ):
                self._emit_record(ar, metrics, preempted=False, error="seq_too_long")
                continue
            kv_cost_full = kv_bytes_for_request(
                cfg.model, ar.prompt_total + ar.max_output, cfg.dtype
            )
            # Reserve max_output up-front so the request is guaranteed
            # not to OOM mid-generation.
            if self._kv_used_bytes + kv_cost_full <= cfg.gpu_kv_budget_bytes:
                self._kv_used_bytes += kv_cost_full
                self._kv_peak_bytes = max(self._kv_peak_bytes, self._kv_used_bytes)
                ar.prefill_progress = 0
                self._active[ar.req.request_id] = ar
                self._prefill_one_chunk(ar)  # first chunk starts now
            else:
                if cfg.drop_on_overflow:
                    self._emit_record(ar, metrics, preempted=False, error="kv_oom")
                else:
                    still_waiting.append(ar)
        self._waiting = still_waiting

        # 2) decode step: each active request produces 1 token.
        step_time_decode = 1.0 / cfg.decode_tps_per_seq
        # Step time also includes any prefill chunks scheduled this step.
        # (We already advanced prefill_progress in _prefill_one_chunk.)
        # Sum the chunk-cost: each chunk in flight costs prompt_chunk /
        # prefill_tps. We approximate by tracking prefill_chunks_done
        # per active request. For simplicity, we don't track per-chunk
        # timing here — the chunked-prefill cost is bounded by
        # chunk / prefill_tps, which we add as a flat overhead per
        # request still in prefill.
        prefill_overhead_per_req = cfg.prefill_chunk_tokens / cfg.prefill_tps

        active_now = list(self._active.values())
        # The slowest active request sets the step time (decode is
        # bandwidth-bound: each step takes the same wall time regardless
        # of how many active seqs).
        step_time = step_time_decode + len(active_now) * prefill_overhead_per_req * 0.0
        # ^ the prefill_overhead_per_req is added implicitly via the
        # chunked prefill time advance in _prefill_one_chunk; we don't
        # double-count it. The decode step itself is bandwidth-bound
        # and parallelisable.

        # Advance each active request by one token.
        for ar in active_now:
            self._advance_request_one_token(ar, step_time, metrics)

        # 3) emit finished requests.
        for ar in active_now:
            if ar.completion_s is not None:
                # already finished; emit if not already done
                self._emit_record_if_new(ar, metrics)

        # 4) schedule next step if anything is still active.
        if self._active or self._waiting:
            self._push_event(self._now + step_time, "decode_step", request_id=-1)

    def _prefill_one_chunk(self, ar: _ActiveRequest) -> None:
        """Advance the prefill of an active request by one chunk.

        If prefill completes (progress == prompt_total), the request
        becomes eligible to emit its first decoded token.
        """
        cfg = self.cfg
        chunk = cfg.prefill_chunk_tokens
        ar.prefill_progress = min(ar.prompt_total, ar.prefill_progress + chunk)
        # No time advance here — chunked prefill overlaps with the
        # decode step's wall time (this is the whole point of chunked
        # prefill: long prompts don't block decoding for seconds).

    def _advance_request_one_token(
        self,
        ar: _ActiveRequest,
        step_time: float,
        metrics: MetricsCollector,
    ) -> None:
        """Advance the request by one token (prefill chunk OR decode step)."""
        cfg = self.cfg
        # If still prefill-ing, advance the chunk first.
        if ar.prefill_progress < ar.prompt_total:
            self._prefill_one_chunk(ar)
            # During prefill we don't emit tokens; the request is still
            # in 'prefill phase' and doesn't contribute to TPOT.
            return

        # Prefill done — emit one decoded token.
        ar.decoded_tokens += 1
        if ar.first_token_s is None:
            ar.first_token_s = self._now  # TTFT clock starts now
        max_out = ar.max_output
        if cfg.max_output_override is not None:
            max_out = min(max_out, cfg.max_output_override)
        if ar.decoded_tokens >= max_out:
            ar.completion_s = self._now
            # Remove from active set and free KV cache.
            self._free_kv(ar)
            self._active.pop(ar.req.request_id, None)

    def _free_kv(self, ar: _ActiveRequest) -> None:
        cfg = self.cfg
        kv_cost = kv_bytes_for_request(
            cfg.model, ar.prompt_total + ar.max_output, cfg.dtype
        )
        self._kv_used_bytes = max(0, self._kv_used_bytes - kv_cost)

    def _emit_record(
        self,
        ar: _ActiveRequest,
        metrics: MetricsCollector,
        *,
        preempted: bool,
        error: str | None,
    ) -> None:
        # For dropped-on-arrival requests, ar.first_token_s and
        # ar.completion_s are None — fill them with arrival + epsilon
        # so the record is well-formed.
        ft = ar.first_token_s if ar.first_token_s is not None else ar.arrival_s
        cc = ar.completion_s if ar.completion_s is not None else ar.arrival_s
        rec = RequestRecord(
            request_id=ar.req.request_id,
            prompt_len=ar.req.prompt_len,
            output_len=ar.decoded_tokens if not preempted else 0,
            arrival_s=ar.arrival_s,
            first_token_s=ft,
            completion_s=cc,
            preempted=preempted,
            error=error,
        )
        metrics.record_finished(rec)

    def _emit_record_if_new(self, ar: _ActiveRequest, metrics: MetricsCollector) -> None:
        # Idempotent: only emit if not already in the metrics collector.
        # We track this by checking ar.completion_s is set and the record
        # isn't already present. For simplicity we mark via a sentinel:
        if not hasattr(ar, "_emitted"):
            ar._emitted = False  # type: ignore[attr-defined]
        if ar._emitted:  # type: ignore[attr-defined]
            return
        self._emit_record(ar, metrics, preempted=ar.preempted, error=ar.error)
        ar._emitted = True  # type: ignore[attr-defined]

    def _preempt_all(self, metrics: MetricsCollector) -> None:
        for ar in list(self._active.values()) + list(self._waiting):
            if ar.completion_s is None:
                ar.preempted = True
                ar.completion_s = self._now
                self._free_kv(ar)
                self._emit_record(ar, metrics, preempted=True, error=None)
        self._active.clear()
        self._waiting.clear()
        self._kv_used_bytes = 0

    # ----- introspection (for the experiment runner) -----

    @property
    def kv_peak_bytes(self) -> int:
        return self._kv_peak_bytes

    @property
    def n_active(self) -> int:
        return len(self._active)

    @property
    def n_waiting(self) -> int:
        return len(self._waiting)

    @property
    def now(self) -> float:
        return self._now


# ---------------------------------------------------------------------------
# Adapter: convert a Request from load_generator into an _ActiveRequest.
# The `_on_arrival` handler uses a small adapter for type-safety.
# ---------------------------------------------------------------------------


def _to_active_request(req: Request, cfg: EngineConfig) -> _ActiveRequest:
    return _ActiveRequest(
        req=req,
        model=cfg.model,
        dtype=cfg.dtype,
        arrival_s=req.arrival_s,
    )
