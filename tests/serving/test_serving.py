"""Tests for load_generator, metrics_collector, simulated_engine."""

from __future__ import annotations

import math

import pytest

from src.token_to_agent.systems.serving import (
    EngineConfig,
    MetricsCollector,
    RequestRecord,
    SimulatedEngine,
    generate_schedule,
    schedule_summary,
)
from src.token_to_agent.systems.serving.load_generator import Request
from src.token_to_agent.systems.serving.metrics_collector import (
    AggregateStats,
    write_summary_csv,
    write_summary_md,
)
from src.token_to_agent.systems.inference.kv_cache import PRESETS


# ---------------------------------------------------------------------------
# load_generator
# ---------------------------------------------------------------------------


def test_generate_schedule_poisson_count():
    reqs = generate_schedule(n_requests=100, rps=10.0, duration_s=100.0, seed=42)
    assert len(reqs) <= 100  # may be less if duration_s caps them
    assert all(isinstance(r, Request) for r in reqs)
    assert all(r.arrival_s >= 0 for r in reqs)


def test_generate_schedule_deterministic():
    a = generate_schedule(n_requests=50, rps=20.0, duration_s=10.0, seed=123)
    b = generate_schedule(n_requests=50, rps=20.0, duration_s=10.0, seed=123)
    assert [(r.request_id, r.arrival_s, r.prompt_len, r.max_output_len) for r in a] == \
           [(r.request_id, r.arrival_s, r.prompt_len, r.max_output_len) for r in b]


def test_generate_schedule_sorted_by_arrival():
    reqs = generate_schedule(n_requests=200, rps=10.0, duration_s=100.0, seed=7)
    arrivals = [r.arrival_s for r in reqs]
    assert arrivals == sorted(arrivals)


def test_generate_schedule_zero_requests():
    assert generate_schedule(n_requests=0, rps=10.0, duration_s=10.0, seed=1) == []


def test_generate_schedule_burst_pattern():
    reqs = generate_schedule(
        n_requests=200, rps=10.0, duration_s=20.0,
        arrival_pattern="burst",
        burst_rps=50.0, burst_start_s=5.0, burst_duration_s=3.0,
        seed=42,
    )
    assert reqs  # non-empty
    # Verify some arrivals fall in the burst window [5, 8).
    in_burst = [r for r in reqs if 5.0 <= r.arrival_s < 8.0]
    in_burst_rate = len(in_burst) / 3.0
    base_rate = (len(reqs) - len(in_burst)) / 17.0
    # With burst_rps=50 and base=10, burst should be ~5x denser.
    # Allow wide margin (small-N Poisson noise).
    assert in_burst_rate > 2 * base_rate


def test_generate_schedule_invalid_pattern():
    with pytest.raises(ValueError):
        generate_schedule(n_requests=10, rps=10.0, duration_s=10.0, arrival_pattern="zzz")


def test_generate_schedule_prompt_capped():
    reqs = generate_schedule(
        n_requests=50, rps=10.0, duration_s=10.0,
        prompt_mean=1000, prompt_max=200, seed=1,
    )
    assert all(r.prompt_len <= 200 for r in reqs)


def test_schedule_summary_keys():
    reqs = generate_schedule(n_requests=30, rps=10.0, duration_s=10.0, seed=1)
    s = schedule_summary(reqs)
    assert s["n"] == 30
    assert s["arrival_span_s"] > 0
    assert s["prompt_mean"] > 0
    assert s["total_input_tokens"] > 0


def test_schedule_summary_empty():
    assert schedule_summary([]) == {"n": 0}


# ---------------------------------------------------------------------------
# metrics_collector
# ---------------------------------------------------------------------------


def _make_record(rid: int, arrival: float, ft: float, completion: float,
                 prompt: int = 100, output: int = 50) -> RequestRecord:
    return RequestRecord(
        request_id=rid,
        prompt_len=prompt,
        output_len=output,
        arrival_s=arrival,
        first_token_s=ft,
        completion_s=completion,
    )


def test_record_ttft_tpot_e2e():
    r = _make_record(0, arrival=0.0, ft=0.5, completion=2.5, output=50)
    assert r.ttft_s == 0.5
    assert r.e2e_s == 2.5
    # TPOT = (2.5 - 0.5) / (50 - 1) = 0.0404
    assert abs(r.tpot_s - (2.0 / 49)) < 1e-9


def test_record_tpot_zero_for_one_token():
    r = _make_record(0, 0.0, 0.1, 0.2, output=1)
    assert r.tpot_s == 0.0


def test_aggregate_quantiles_basic():
    s = AggregateStats(label="TTFT", unit="ms")
    for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
        s.add(v)
    assert s.p50 == 30.0
    assert s.max == 50.0
    assert s.min == 10.0
    assert abs(s.mean - 30.0) < 1e-9


def test_aggregate_quantiles_interpolation():
    # Linear interpolation (type 7) at n=4, q=0.5 -> sample 1.5
    s = AggregateStats(label="X", unit="ms")
    for v in [0, 1, 2, 3]:
        s.add(v)
    # pos = 0.5 * 3 = 1.5; interpolate between sample 1 (val=1) and 2 (val=2)
    assert s.p50 == 1.5


def test_aggregate_empty_is_zero():
    s = AggregateStats(label="X", unit="ms")
    assert s.p50 == 0.0
    assert s.max == 0.0
    assert s.count == 0


def test_metrics_collector_summarise_basic():
    m = MetricsCollector()
    for i, (a, ft, c, out) in enumerate([
        (0.0, 0.1, 0.6, 50),
        (0.0, 0.2, 0.7, 50),
        (0.0, 0.3, 0.8, 50),
    ]):
        m.record_finished(_make_record(i, a, ft, c, output=out))
    s = m.summarise(wallclock_s=1.0, cost_per_hour_usd=2.0)
    assert s["n_total_requests"] == 3
    assert s["n_completed"] == 3
    assert s["TPOT_count"] == 3
    assert s["TTFT_p50_ms"] > 0
    assert s["E2E_p50_ms"] > 0
    assert s["cost_usd_total"] > 0
    assert s["cost_usd_per_1m_output"] > 0
    assert s["cost_usd_per_successful_task"] > 0


def test_metrics_collector_preempted_and_errors():
    m = MetricsCollector()
    m.record_finished(_make_record(0, 0, 0.1, 0.2, output=10))
    m.record_finished(RequestRecord(1, 100, 0, 0.0, 0.0, 0.0, preempted=True))
    m.record_finished(RequestRecord(2, 100, 0, 0.0, 0.0, 0.0, error="kv_oom"))
    assert m.n_completed == 1
    assert m.n_preempted == 1
    assert m.n_errors == 1


def test_write_summary_csv_and_md():
    rows = [
        {"a": 1.0, "b": 2, "c": "x"},
        {"a": 3.0, "b": 4, "c": "y"},
    ]
    csv_out = write_summary_csv(rows)
    assert "a,b,c" in csv_out
    assert "1.0,2,x" in csv_out
    md_out = write_summary_md(rows)
    assert "| a | b | c |" in md_out
    assert "| 1 | 2 | x |" in md_out


def test_write_summary_empty():
    assert write_summary_csv([]) == ""
    assert write_summary_md([]) == "_no data_\n"


# ---------------------------------------------------------------------------
# simulated_engine
# ---------------------------------------------------------------------------


def _small_cfg() -> EngineConfig:
    """Engine config with a generous budget so small workloads all fit."""
    return EngineConfig(
        model=PRESETS["mistral-7b"],  # GQA, smaller per-token KV
        gpu_kv_budget_bytes=10 * 1024**3,
        decode_tps_per_seq=80.0,
        prefill_tps=8000.0,
        prefill_chunk_tokens=256,
        max_output_override=64,
    )


def test_engine_runs_small_workload_completes_all():
    reqs = generate_schedule(n_requests=10, rps=5.0, duration_s=10.0, seed=42)
    cfg = _small_cfg()
    m = MetricsCollector()
    eng = SimulatedEngine(cfg)
    eng.run(reqs, m)
    assert m.n_completed == 10
    assert m.n_preempted == 0
    assert m.n_errors == 0


def test_engine_admits_based_on_kv_budget():
    """If budget is too small, late requests should be dropped (kv_oom)."""
    cfg = EngineConfig(
        model=PRESETS["llama2-7b"],  # MHA = large KV
        gpu_kv_budget_bytes=1 * 1024**3,  # 1 GB total — way too small
        decode_tps_per_seq=80.0,
        prefill_tps=8000.0,
        prefill_chunk_tokens=256,
        max_output_override=64,
        drop_on_overflow=True,
    )
    # 50 requests of 1024-prompt + 64-out => 1024+64 = 1088 tokens each
    # => ~540 KB KV per request (mistral) or ~1.1 MB (llama-2 MHA).
    # With 1 GB budget, llama2 can fit ~900 requests of 1088 tokens —
    # actually too many. Try tighter.
    cfg.gpu_kv_budget_bytes = 5 * 1024**2  # 5 MB
    reqs = generate_schedule(n_requests=50, rps=20.0, duration_s=10.0, seed=42)
    m = MetricsCollector()
    eng = SimulatedEngine(cfg)
    eng.run(reqs, m)
    # At least some requests should be dropped or preempted.
    assert m.n_errors + m.n_preempted > 0


def test_engine_rejects_oversized_request():
    cfg = EngineConfig(
        model=PRESETS["llama2-7b"],
        gpu_kv_budget_bytes=60 * 1024**3,
        decode_tps_per_seq=80.0,
        prefill_tps=8000.0,
        prefill_chunk_tokens=256,
        max_output_override=4096,
        enforce_max_seq_len=True,
    )
    # 4096-prompt + 4096-output = 8192 > 4096 max_seq_len.
    reqs = [Request(request_id=0, arrival_s=0.0, prompt_len=4096, max_output_len=4096)]
    m = MetricsCollector()
    SimulatedEngine(cfg).run(reqs, m)
    assert m.n_errors == 1
    assert m.records[0].error == "seq_too_long"


def test_engine_decode_tpot_matches_config():
    cfg = _small_cfg()
    # 1 request, 32 output tokens.
    reqs = [Request(request_id=0, arrival_s=0.0, prompt_len=64, max_output_len=32)]
    m = MetricsCollector()
    SimulatedEngine(cfg).run(reqs, m)
    expected_tpot_ms = 1000.0 / cfg.decode_tps_per_seq  # 12.5 ms
    assert abs(m.records[0].tpot_s * 1000.0 - expected_tpot_ms) < 1e-3


def test_engine_no_preempt_when_within_budget():
    cfg = _small_cfg()
    reqs = generate_schedule(n_requests=5, rps=2.0, duration_s=10.0, seed=42)
    m = MetricsCollector()
    eng = SimulatedEngine(cfg)
    eng.run(reqs, m)
    assert m.n_preempted == 0
    assert eng.kv_peak_bytes >= 0  # peak tracker wired up
