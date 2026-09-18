"""W5 — exp-003 serving benchmark (CPU-runnable).

Sweeps 4 variables against the simulated engine:
  - rps (effective offered load)
  - prompt_len
  - gen_len
  - arrival_pattern (poisson | burst)

Records per cell:
  - TTFT p50 / p95 / max
  - TPOT p50 / p95
  - E2E p50 / p95
  - throughput tokens/s and req/s
  - preempted / dropped counts
  - peak KV cache usage (bytes)

Outputs:
  - results/serving.csv
  - results/serving.md
  - results/serving.json

Run on the Jetson (CPU) for scheduler-logic validation. For real
vLLM numbers, see experiments/w05/README-JETSON-LIMITATIONS.md.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.token_to_agent.systems.inference.kv_cache import (
    PRESETS,
    cost_per_million_tokens,
    max_concurrent_at_seq_len,
)
from src.token_to_agent.systems.serving import (
    EngineConfig,
    MetricsCollector,
    SimulatedEngine,
    generate_schedule,
    schedule_summary,
)


@dataclass
class SweepCell:
    rps: float
    prompt_len: int
    gen_len: int
    arrival_pattern: str
    n_requests: int
    duration_s: float
    seed: int


# Default sweep grid. Keep small so the simulator runs in ~30s on Jetson.
# prompt_len + gen_len <= 4096 (Llama-2-7B max_seq_len).
DEFAULT_GRID: list[SweepCell] = []
for rps in [4.0, 16.0, 64.0]:
    for prompt_len in [256, 1024, 2048]:
        for gen_len in [64, 256]:
            # Skip 2048+256 = 2304 (fine) and 2048+64 (fine).
            # Skip pairs where prompt_len >= 4096 - gen_len (would be seq_too_long).
            if prompt_len + gen_len > 4096:
                continue
            DEFAULT_GRID.append(
                SweepCell(
                    rps=rps,
                    prompt_len=prompt_len,
                    gen_len=gen_len,
                    arrival_pattern="poisson",
                    n_requests=200,
                    duration_s=200.0,
                    seed=42,
                )
            )


def _run_one(cell: SweepCell, model_name: str = "llama2-7b") -> dict:
    """Run a single sweep cell and return the summary dict."""
    cfg_model = PRESETS[model_name]
    # Pick a KV budget that comfortably fits the workload. For the
    # simulated engine this is just the admission cap; real GPU
    # memory is set to 60 GB to leave room for weights + activations.
    cfg = EngineConfig(
        model=cfg_model,
        dtype="fp16",
        gpu_kv_budget_bytes=60 * 1024**3,
        decode_tps_per_seq=50.0,  # H100 decode ~50 tokens/s for 7B-class
        prefill_tps=8000.0,
        prefill_chunk_tokens=256,
        max_output_override=cell.gen_len,
        drop_on_overflow=True,
    )
    requests = generate_schedule(
        n_requests=cell.n_requests,
        rps=cell.rps,
        duration_s=cell.duration_s,
        prompt_mean=cell.prompt_len,
        prompt_max=cell.prompt_len * 2,
        output_mean=cell.gen_len,
        output_max=cell.gen_len,
        arrival_pattern=cell.arrival_pattern,
        seed=cell.seed,
    )
    work_summary = schedule_summary(requests)

    metrics = MetricsCollector()
    engine = SimulatedEngine(cfg)
    t0 = time.monotonic()
    engine.run(requests, metrics)
    wallclock = time.monotonic() - t0

    summary = metrics.summarise(
        wallclock_s=engine.now,
        gpu_memory_peak_gb=engine.kv_peak_bytes / 1024**3,
        cost_per_hour_usd=2.0,  # H100 on-demand reference
    )

    # Combine workload + summary into one row.
    row = {
        "model": model_name,
        "rps": cell.rps,
        "prompt_len_target": cell.prompt_len,
        "gen_len_target": cell.gen_len,
        "arrival_pattern": cell.arrival_pattern,
        "n_requests_target": cell.n_requests,
        "n_requests_actual": work_summary.get("n", 0),
        "arrival_span_s": work_summary.get("arrival_span_s", 0.0),
        "total_input_tokens": work_summary.get("total_input_tokens", 0),
        "total_max_output_tokens": work_summary.get("total_max_output_tokens", 0),
        "wallclock_sim_s": engine.now,
        "wallclock_cpu_s": wallclock,
        "kv_peak_gb": summary.get("gpu_memory_peak_gb", 0.0),
        "n_completed": summary.get("n_completed", 0),
        "n_preempted": summary.get("n_preempted", 0),
        "n_errors": summary.get("n_errors", 0),
        "TTFT_p50_ms": summary.get("TTFT_p50_ms", 0.0),
        "TTFT_p95_ms": summary.get("TTFT_p95_ms", 0.0),
        "TTFT_max_ms": summary.get("TTFT_max_ms", 0.0),
        "TPOT_p50_ms": summary.get("TPOT_p50_ms", 0.0),
        "TPOT_p95_ms": summary.get("TPOT_p95_ms", 0.0),
        "TPOT_mean_ms": summary.get("TPOT_mean_ms", 0.0),
        "E2E_p50_ms": summary.get("E2E_p50_ms", 0.0),
        "E2E_p95_ms": summary.get("E2E_p95_ms", 0.0),
        "E2E_max_ms": summary.get("E2E_max_ms", 0.0),
        "throughput_input_tps": summary.get("throughput_input_tps", 0.0),
        "throughput_output_tps": summary.get("throughput_output_tps", 0.0),
        "throughput_rps": summary.get("throughput_requests_per_s", 0.0),
        "cost_usd_per_1m_input": summary.get("cost_usd_per_1m_input", 0.0),
        "cost_usd_per_1m_output": summary.get("cost_usd_per_1m_output", 0.0),
        "cost_usd_per_successful_task": summary.get("cost_usd_per_successful_task", 0.0),
    }
    return row


def main(grid: list[SweepCell] | None = None) -> None:
    grid = grid or DEFAULT_GRID
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_start = time.monotonic()
    for i, cell in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] rps={cell.rps} prompt={cell.prompt_len} gen={cell.gen_len} pattern={cell.arrival_pattern} ...", flush=True)
        row = _run_one(cell)
        rows.append(row)
        print(
            f"    completed={row['n_completed']} preempted={row['n_preempted']} "
            f"errors={row['n_errors']} "
            f"TTFT_p95={row['TTFT_p95_ms']:.1f}ms TPOT_p50={row['TPOT_p50_ms']:.1f}ms "
            f"E2E_p95={row['E2E_p95_ms']:.1f}ms out_tps={row['throughput_output_tps']:.1f}",
            flush=True,
        )
    print(f"\nTotal wallclock (CPU sim): {time.monotonic() - t_start:.2f}s over {len(grid)} cells")

    # CSV
    csv_path = out_dir / "serving.csv"
    cols = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # JSON
    json_path = out_dir / "serving.json"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {json_path}")

    # MD summary
    md_path = out_dir / "serving.md"
    with md_path.open("w") as f:
        f.write(f"# W5 exp-003 — Serving benchmark\n\n")
        f.write(f"Model: `llama2-7b` (32L, 32H, 32 KV, D=128, fp16)\n")
        f.write(f"Sim: discrete-event CPU simulator (`src/token_to_agent/systems/serving/simulated_engine.py`)\n")
        f.write(f"Decode TPS-per-seq: 50.0; prefill TPS: 8000; chunk: 256 tokens\n")
        f.write(f"GPU KV budget: 60 GB (CPU sim only — see JETSON-LIMITATIONS for real GPU runs)\n")
        f.write(f"Cost reference: $2.00/hr H100 on-demand\n\n")
        f.write("## Sweep results\n\n")
        f.write("| rps | prompt | gen | n_completed | preempted | errors | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | E2E p95 (ms) | out TPS | $/1M in | $/1M out |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['rps']} | {r['prompt_len_target']} | {r['gen_len_target']} | "
                f"{r['n_completed']} | {r['n_preempted']} | {r['n_errors']} | "
                f"{r['TTFT_p50_ms']:.1f} | {r['TTFT_p95_ms']:.1f} | "
                f"{r['TPOT_p50_ms']:.1f} | {r['E2E_p95_ms']:.1f} | "
                f"{r['throughput_output_tps']:.1f} | "
                f"{r['cost_usd_per_1m_input']:.3f} | {r['cost_usd_per_1m_output']:.3f} |\n"
            )
        f.write("\n## Notes\n")
        f.write("- All times are simulated wall-clock; see `src/token_to_agent/systems/serving/simulated_engine.py`.\n")
        f.write("- Chunked prefill overlaps decode steps; prefill wall-time is **not** charged per step (see limitations).\n")
        f.write("- Decode TPS-per-seq is the assumed per-sequence decode speed; calibrate on real GPU for absolute numbers.\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
