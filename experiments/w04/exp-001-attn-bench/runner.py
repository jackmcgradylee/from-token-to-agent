"""W4 — Attention benchmark harness (CPU-runnable; CUDA path stubbed).

Measures forward + forward+backward latency, peak memory, and throughput
for the four declared backends (naive / pytorch / triton / reference) at
configurable (B, H, H_kv, T, D).

Hardware notes:
  - On a CUDA host, all four backends run; Triton row reports the real
    FlashAttention-style speedup vs SDPA / naive.
  - On a CUDA-less host (this Jetson), the triton row falls back to
    `reference` via `interface.call`, and timing numbers are reported
    with a `cuda_available=False` flag. The harness still produces a CSV
    with `forward_ms` / `fwd_bwd_ms` / `peak_mem_mb` rows so the artifact
    can be regenerated when a GPU host is available.

What we measure:
  - `forward_ms`   : wall time for one forward pass (after warmup).
  - `fwd_bwd_ms`   : wall time for forward+backward (one autograd.step).
  - `peak_mem_mb`  : peak resident-set-size delta attributable to the
                    forward pass (process-level RSS, via /proc/self/statm
                    since torch.cuda.max_memory_allocated needs CUDA).
  - `correctness`  : max abs error vs the reference output.
  - `tflops`       : effective FLOPs achieved (T*Q@K^T + A@V, the two
                    matmuls) divided by forward_ms.

Outputs:
  - results/bench.csv   one row per (backend, B, H, H_kv, T, D).
  - results/bench.md    human-readable summary table.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.token_to_agent.from_scratch.kernels import (
    interface as attn_iface,
    reference_attention,
    triton_attention as triton_mod,
)

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"

# Default sweep grid (B, H, H_kv, T, D).
DEFAULT_GRID: list[tuple[int, int, int, int, int]] = [
    (1, 4, 4, 64, 32),
    (1, 4, 4, 128, 32),
    (1, 8, 2, 256, 64),
    (1, 8, 2, 512, 64),
    (2, 8, 2, 1024, 64),
]

BACKENDS_TO_BENCH = ["naive", "reference", "pytorch", "triton"]
WARMUP = 3
REPEAT = 5


@dataclass
class BenchRow:
    backend: str
    B: int
    H: int
    H_kv: int
    T: int
    D: int
    cuda_available: bool
    forward_ms: float
    fwd_bwd_ms: float
    peak_mem_mb: float
    max_abs_err: float
    tflops: float


def _rss_mb() -> float:
    """Return resident-set-size in MB via /proc/self/statm."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            pages = int(f.read().split()[1])  # resident pages
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024
    except Exception:
        return 0.0


def _tflops(B: int, H: int, T: int, D: int, ms: float) -> float:
    """FLOPs for a single forward pass = 2 * B*H * T^2 * D + 2 * B*H * T * T * D
    (the two matmuls Q@K^T and A@V). Returns TFLOPs achieved."""
    if ms <= 0:
        return 0.0
    flops = 2 * B * H * T * T * D + 2 * B * H * T * T * D  # the two matmuls
    flops_per_s = flops / (ms / 1000.0)
    return flops_per_s / 1e12


def _bench_one(backend: str, B: int, H: int, H_kv: int, T: int, D: int,
               *, seed: int = 0, warmup: int = WARMUP, repeat: int = REPEAT,
               ) -> BenchRow:
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, D, dtype=torch.float32) * 0.5
    k = torch.randn(B, H_kv, T, D, dtype=torch.float32) * 0.5
    v = torch.randn(B, H_kv, T, D, dtype=torch.float32) * 0.5
    n_rep = H // H_kv
    cuda_available = torch.cuda.is_available()

    # Reference output for correctness.
    ref = reference_attention.reference_attention_forward(
        q, k, v, is_causal=True, n_rep=n_rep,
    )

    def _fwd() -> torch.Tensor:
        return attn_iface.call(backend, q, k, v, is_causal=True, n_rep=n_rep)

    def _fwd_bwd() -> torch.Tensor:
        qa = q.detach().clone().requires_grad_(True)
        ka = k.detach().clone().requires_grad_(True)
        va = v.detach().clone().requires_grad_(True)
        out = attn_iface.call(backend, qa, ka, va, is_causal=True, n_rep=n_rep)
        out.sum().backward()
        return out

    # Warmup.
    for _ in range(warmup):
        _fwd()
        _fwd_bwd()

    # Timed forward.
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    out = _fwd()
    forward_ms = (time.perf_counter() - t0) * 1000.0
    rss_after = _rss_mb()
    peak_mem = max(0.0, rss_after - rss_before)

    # Timed forward+backward.
    t0 = time.perf_counter()
    _fwd_bwd()
    fwd_bwd_ms = (time.perf_counter() - t0) * 1000.0

    # Correctness (max abs err vs reference).
    max_err = (out - ref).abs().max().item()

    # Repeat to take median of `repeat` runs.
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _fwd()
        times.append((time.perf_counter() - t0) * 1000.0)
    forward_ms = sorted(times)[len(times) // 2]

    return BenchRow(
        backend=backend,
        B=B, H=H, H_kv=H_kv, T=T, D=D,
        cuda_available=cuda_available,
        forward_ms=forward_ms,
        fwd_bwd_ms=fwd_bwd_ms,
        peak_mem_mb=peak_mem,
        max_abs_err=max_err,
        tflops=_tflops(B, H, T, D, forward_ms),
    )


def main(grid: Optional[list[tuple[int, int, int, int, int]]] = None) -> int:
    grid = grid or DEFAULT_GRID
    RESULTS_DIR.mkdir(exist_ok=True)

    rows: list[BenchRow] = []
    for shape in grid:
        for backend in BACKENDS_TO_BENCH:
            try:
                row = _bench_one(backend, *shape)
            except Exception as e:  # noqa: BLE001
                print(f"[exp-001] skip {backend} {shape}: {e!r}")
                continue
            rows.append(row)
            print(
                f"[exp-001] {backend:9s} B={row.B} H={row.H} H_kv={row.H_kv} "
                f"T={row.T} D={row.D}  fwd={row.forward_ms:8.3f}ms "
                f"fwd+bwd={row.fwd_bwd_ms:8.3f}ms  "
                f"mem={row.peak_mem_mb:6.1f}MB  err={row.max_abs_err:.2e}  "
                f"{row.tflops:6.3f}TFLOPS  cuda={row.cuda_available}"
            )

    # CSV.
    csv_path = RESULTS_DIR / "bench.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    # JSON (same content).
    json_path = RESULTS_DIR / "bench.json"
    json_path.write_text(
        json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Markdown summary.
    md_path = RESULTS_DIR / "bench.md"
    md = ["# W4 exp-001 — Attention backend benchmark\n"]
    md.append(f"Grid: {grid!r}\n")
    md.append(f"CUDA available: **{torch.cuda.is_available()}**\n")
    md.append("| backend | B | H | H_kv | T | D | fwd (ms) | fwd+bwd (ms) | "
              "peak mem (MB) | max err | TFLOPS |\n")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
              "---: | ---: | ---: |\n")
    for r in rows:
        md.append(
            f"| `{r.backend}` | {r.B} | {r.H} | {r.H_kv} | {r.T} | {r.D} | "
            f"{r.forward_ms:.3f} | {r.fwd_bwd_ms:.3f} | {r.peak_mem_mb:.1f} | "
            f"{r.max_abs_err:.2e} | {r.tflops:.3f} |\n"
        )
    md.append("\n## Notes\n")
    md.append("- All times are median of 5 runs after 3 warmup iterations.\n")
    md.append("- `peak_mem_mb` is process-level RSS delta around the forward pass; "
              "for CUDA we'd switch to `torch.cuda.max_memory_allocated`.\n")
    md.append("- `max_abs_err` is the max absolute difference vs `reference` (fp32).\n")
    md.append("- On a CUDA-less host, the `triton` row falls back to `reference` "
              "via the interface; rerun on a GPU host for a real speedup number.\n")
    md_path.write_text("".join(md), encoding="utf-8")

    print(f"\n[exp-001] wrote {csv_path} ({len(rows)} rows)")
    print(f"[exp-001] wrote {json_path}")
    print(f"[exp-001] wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())