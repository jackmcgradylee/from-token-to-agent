"""W4 — Attention seq-len scaling sweep (exp-002).

Sweeps sequence length on the four backends to show how latency grows
with T. The classic "naive attention is O(T^2)" plot.

Outputs:
  - results/seqlen.csv     rows: (backend, T, fwd_ms, fwd_bwd_ms, tflops)
  - results/seqlen.png     log-log latency vs T
  - results/seqlen.md      markdown report

Hardware notes:
  - On CUDA: the Triton row should show a sub-quadratic or near-linear
    region for large T (Flash-style tiling).
  - On CPU: Triton falls back to reference; the curve shows the naive
    matmul cost. Rerun on a GPU host to see the Flash effect.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.token_to_agent.from_scratch.kernels import (
    interface as attn_iface,
    reference_attention,
)

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "results"

# Shapes are (T, batch, n_heads, n_kv_heads, head_dim) for readability in
# the CSV header; `_bench` takes (B, H, H_kv, T, D) so we reorder.
SHAPES_RAW: list[tuple[int, int, int, int, int]] = [
    (128,  1, 8, 2, 64),
    (256,  1, 8, 2, 64),
    (512,  1, 8, 2, 64),
    (1024, 1, 8, 2, 64),
    (2048, 1, 8, 2, 64),
]


def _shape_for_bench(t: int, batch: int, h: int, h_kv: int, d: int) -> tuple[int, int, int, int, int]:
    return (batch, h, h_kv, t, d)

BACKENDS = ["naive", "reference", "pytorch", "triton"]
WARMUP = 1
REPEAT = 2


@dataclass
class SeqLenRow:
    backend: str
    T: int
    B: int
    H: int
    H_kv: int
    D: int
    forward_ms: float
    fwd_bwd_ms: float
    tflops: float


def _bench(backend: str, B: int, H: int, H_kv: int, T: int, D: int) -> SeqLenRow:
    torch.manual_seed(T)
    q = torch.randn(B, H, T, D, dtype=torch.float32) * 0.5
    k = torch.randn(B, H_kv, T, D, dtype=torch.float32) * 0.5
    v = torch.randn(B, H_kv, T, D, dtype=torch.float32) * 0.5
    n_rep = H // H_kv

    def _fwd() -> torch.Tensor:
        return attn_iface.call(backend, q, k, v, is_causal=True, n_rep=n_rep)

    def _fwd_bwd() -> torch.Tensor:
        qa = q.detach().clone().requires_grad_(True)
        ka = k.detach().clone().requires_grad_(True)
        va = v.detach().clone().requires_grad_(True)
        out = attn_iface.call(backend, qa, ka, va, is_causal=True, n_rep=n_rep)
        out.sum().backward()

    for _ in range(WARMUP):
        _fwd()
        _fwd_bwd()

    # Median of REPEAT runs (use forward only for the seq-len curve; the
    # fwd+bwd curve is also reported).
    fwd_times, bwd_times = [], []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        _fwd()
        fwd_times.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        _fwd_bwd()
        bwd_times.append((time.perf_counter() - t0) * 1000.0)

    forward_ms = sorted(fwd_times)[len(fwd_times) // 2]
    fwd_bwd_ms = sorted(bwd_times)[len(bwd_times) // 2]

    flops = 4 * B * H * T * T * D  # both matmuls combined
    tflops = (flops / (forward_ms / 1000.0)) / 1e12 if forward_ms > 0 else 0.0
    return SeqLenRow(backend=backend, T=T, B=B, H=H, H_kv=H_kv, D=D,
                     forward_ms=forward_ms, fwd_bwd_ms=fwd_bwd_ms, tflops=tflops)


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows: list[SeqLenRow] = []
    for shape_raw in SHAPES_RAW:
        T_raw, B, H, H_kv, D = shape_raw
        shape = _shape_for_bench(T_raw, B, H, H_kv, D)
        for backend in BACKENDS:
            try:
                row = _bench(backend, *shape)
            except Exception as e:  # noqa: BLE001
                print(f"[exp-002] skip {backend} {shape}: {e!r}")
                continue
            rows.append(row)
            print(
                f"[exp-002] {backend:9s} T={row.T:5d}  "
                f"fwd={row.forward_ms:8.2f}ms  fwd+bwd={row.fwd_bwd_ms:8.2f}ms  "
                f"{row.tflops:6.3f}TFLOPS"
            )

    # CSV
    csv_path = RESULTS_DIR / "seqlen.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    # Plot (log-log)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for backend, color in zip(BACKENDS, ["C0", "C1", "C2", "C3"]):
            sub = [r for r in rows if r.backend == backend]
            sub.sort(key=lambda r: r.T)
            xs = [r.T for r in sub]
            fwd = [r.forward_ms for r in sub]
            bwd = [r.fwd_bwd_ms for r in sub]
            axes[0].loglog(xs, fwd, marker="o", label=backend, color=color)
            axes[1].loglog(xs, bwd, marker="s", label=backend, color=color)
        for ax, title in zip(axes, ["forward (ms)", "forward+backward (ms)"]):
            ax.set_xlabel("T (seq length)")
            ax.set_ylabel(title)
            ax.set_title(f"W4 seq-len sweep — {title}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "seqlen.png", dpi=110)
        plt.close()
    except ImportError:
        pass

    # Markdown summary
    md_path = RESULTS_DIR / "seqlen.md"
    md = ["# W4 exp-002 — seq-len sweep\n"]
    md.append(f"CUDA available: **{torch.cuda.is_available()}**\n")
    md.append("| backend | T | fwd (ms) | fwd+bwd (ms) | TFLOPS |\n")
    md.append("| --- | ---: | ---: | ---: | ---: |\n")
    for r in rows:
        md.append(
            f"| `{r.backend}` | {r.T} | {r.forward_ms:.2f} | {r.fwd_bwd_ms:.2f} | {r.tflops:.3f} |\n"
        )
    md.append("\n## Plot\n")
    md.append("`results/seqlen.png` — log-log latency vs T, both forward and forward+backward.\n")
    md.append("\n## Notes\n")
    md.append("- Naive PyTorch attention has O(T^2) matmul cost; latency should grow roughly "
              "as a line with slope 2 on the log-log plot.\n")
    md.append("- Flash-style tiled kernels achieve O(T) effective bandwidth by never "
              "materialising the full (T, T) matrix; the Triton row on a CUDA host "
              "should show a flatter curve at large T.\n")
    md.append("- On a CUDA-less host the Triton row falls back to `reference`; rerun "
              "on a GPU host to see the Flash effect.\n")
    md_path.write_text("".join(md), encoding="utf-8")

    print(f"\n[exp-002] wrote {csv_path} ({len(rows)} rows)")
    print(f"[exp-002] wrote {RESULTS_DIR / 'seqlen.png'}")
    print(f"[exp-002] wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())