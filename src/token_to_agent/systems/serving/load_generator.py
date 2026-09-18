"""Synthetic load generator for serving benchmarks.

Generates a schedule of `Request` objects with arrival times and
prompt/output lengths, drawn from a configurable distribution. The
simulated engine consumes the schedule instead of waiting in real
time, which makes the benchmark deterministic and CI-friendly.

Three arrival patterns:
- "poisson": exponential inter-arrival, mean rate = `rps` requests/sec.
- "burst":   `rps` Poisson until time T_burst, then a sudden step to
             `burst_rps` for `burst_duration_s`, then back to `rps`.
             Mimics a viral moment / a launch-day traffic spike.
- "trace":   a list of inter-arrival deltas and (prompt_len, max_out)
             tuples. Reserved for replaying real production traces
             (e.g. Azure LLM serving trace). Not implemented here;
             the API is fixed so the engine can call either.

Length distributions:
- prompt_len: lognormal-ish, mean = `prompt_mean`, capped by
  `prompt_max`. Roughly matches chat workload shapes.
- max_output_len: similar, mean = `output_mean`, cap = `output_max`.

Determinism: `seed` (int) is forwarded to a private `random.Random`
so two runs with the same seed produce identical schedules.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_s: float
    prompt_len: int
    max_output_len: int


def _lognormal(rng: random.Random, mean: float, sigma: float = 0.6) -> float:
    """Lognormal with given *mean* (in linear units), not log-mean.

    mu_log = log(mean) - sigma^2 / 2  so E[X] = mean.
    """
    if mean <= 0:
        return 0.0
    mu_log = math.log(mean) - 0.5 * sigma * sigma
    return math.exp(rng.gauss(mu_log, sigma))


def generate_schedule(
    *,
    n_requests: int,
    rps: float,
    duration_s: float | None = None,
    prompt_mean: int = 512,
    prompt_max: int = 4096,
    output_mean: int = 256,
    output_max: int = 1024,
    arrival_pattern: str = "poisson",
    burst_rps: float | None = None,
    burst_start_s: float | None = None,
    burst_duration_s: float | None = None,
    seed: int = 42,
) -> list[Request]:
    """Return a list of `Request`s ordered by `arrival_s`.

    Parameters are validated; bad combinations raise ValueError.

    For "burst", if burst_rps / burst_start_s / burst_duration_s are
    not all provided, defaults are filled:
        burst_rps       = 5 × rps
        burst_start_s   = duration_s / 3
        burst_duration_s= duration_s / 5
    """
    if arrival_pattern not in ("poisson", "burst"):
        raise ValueError(f"unknown arrival_pattern {arrival_pattern!r}")
    if rps <= 0:
        raise ValueError("rps must be > 0")
    if n_requests <= 0:
        return []
    if duration_s is None:
        # Default duration: enough to fit n_requests at rps, with 50% headroom.
        duration_s = n_requests / rps * 1.5
    if arrival_pattern == "burst":
        burst_rps = burst_rps or (5 * rps)
        burst_start_s = burst_start_s if burst_start_s is not None else duration_s / 3
        burst_duration_s = (
            burst_duration_s if burst_duration_s is not None else duration_s / 5
        )
        if burst_start_s + burst_duration_s > duration_s:
            raise ValueError("burst window exceeds duration_s")
        if burst_rps <= rps:
            raise ValueError("burst_rps should exceed base rps to be a 'burst'")

    rng = random.Random(seed)
    requests: list[Request] = []

    # 1) arrival times
    arrivals: list[float] = []
    t = 0.0
    while len(arrivals) < n_requests:
        if arrival_pattern == "poisson":
            delta = rng.expovariate(rps)
            t += delta
        else:  # burst
            in_burst = burst_start_s <= t < burst_start_s + burst_duration_s  # type: ignore[operator]
            rate = burst_rps if in_burst else rps  # type: ignore[operator]
            t += rng.expovariate(rate)
        if t > duration_s:
            break  # cap to duration_s
        arrivals.append(t)

    # 2) prompt / output lengths (independent draws)
    for rid, a in enumerate(arrivals):
        p = int(min(prompt_max, max(1, round(_lognormal(rng, prompt_mean)))))
        o = int(min(output_max, max(1, round(_lognormal(rng, output_mean)))))
        requests.append(
            Request(request_id=rid, arrival_s=a, prompt_len=p, max_output_len=o)
        )

    return requests


def schedule_summary(requests: list[Request]) -> dict:
    """Convenience summary used by experiments/w05 exp-003 to print
    a one-liner about the workload before running the engine."""
    if not requests:
        return {"n": 0}
    arrivals = [r.arrival_s for r in requests]
    prompts = [r.prompt_len for r in requests]
    outputs = [r.max_output_len for r in requests]
    return {
        "n": len(requests),
        "arrival_first_s": min(arrivals),
        "arrival_last_s": max(arrivals),
        "arrival_span_s": max(arrivals) - min(arrivals),
        "prompt_mean": sum(prompts) / len(prompts),
        "prompt_max": max(prompts),
        "output_mean": sum(outputs) / len(outputs),
        "output_max": max(outputs),
        "total_input_tokens": sum(prompts),
        "total_max_output_tokens": sum(outputs),
    }
