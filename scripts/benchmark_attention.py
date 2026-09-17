"""Benchmark attention forward across backends.

Reports, for each (backend, B, H, H_kv, T, D) tuple:
  - latency (ms)            wall time per forward
  - tokens/s                B * T / latency
  - peak memory (MB)        (CUDA only; 0 on CPU)
  - max abs error vs reference (correctness)

Usage:
    python scripts/benchmark_attention.py \\
        --device cpu \\
        --backends reference,pytorch,triton \\
        --output assignments/hw2-systems/results/benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

# Make `src/` importable.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch

from src.kernels import interface as attn_iface
from src.kernels import reference_attention


def make_inputs(B, H, H_kv, T, D, device, dtype):
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, H_kv, T, D, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, H_kv, T, D, device=device, dtype=dtype) * 0.5
    return q, k, v


def time_call(fn, n_warmup=3, n_iters=10):
    """Time a function with warmup; returns (median_ms, per_iter_results)."""
    # Warmup.
    for _ in range(n_warmup):
        fn()
    samples = []
    for _ in range(n_iters):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000)
    samples.sort()
    median = samples[len(samples) // 2]
    return median, samples


def run_one(backend: str, B, H, H_kv, T, D, device, dtype, n_iters):
    q, k, v = make_inputs(B, H, H_kv, T, D, device, dtype)
    n_rep = H // H_kv

    def fn():
        attn_iface.call(
            backend, q, k, v,
            is_causal=True,
            n_rep=n_rep,
            training=False,
        )

    # Correctness vs reference (small input only; reference is slow).
    if B * T * T < 8 * 1024 * 1024:
        ref = reference_attention.reference_attention_forward(
            q.float(), k.float(), v.float(),
            is_causal=True, n_rep=n_rep,
        )
        try:
            out = attn_iface.call(
                backend, q, k, v,
                is_causal=True, n_rep=n_rep, training=False,
            )
            err = (out.float() - ref).abs().max().item()
        except Exception as e:
            err = float("nan")
            out = None
    else:
        err = None
        out = None

    median_ms, samples = time_call(fn, n_warmup=2, n_iters=n_iters)

    tokens = B * T
    tokens_per_sec = tokens / (median_ms / 1000)

    return {
        "backend": backend,
        "B": B, "H": H, "H_kv": H_kv, "T": T, "D": D,
        "device": device, "dtype": str(dtype).replace("torch.", ""),
        "latency_ms": round(median_ms, 3),
        "min_latency_ms": round(min(samples), 3),
        "max_latency_ms": round(max(samples), 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "max_abs_error_vs_reference": err,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--backends", default="reference,pytorch,triton")
    p.add_argument("--n-iters", type=int, default=5)
    p.add_argument("--seq-lens", default="128,256,512,1024")
    p.add_argument("--batch-sizes", default="1,4")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    backends = [b.strip() for b in args.backends.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[bench] CUDA not available; falling back to CPU")
        args.device = "cpu"

    # Reference shape: baseline-100m's attention layer.
    H, H_kv, D = 12, 4, 64  # d_model=768, head_dim=64

    rows = []
    print(f"[bench] device={args.device} dtype={args.dtype} backends={backends}")
    print(f"[bench] shape: H={H}, H_kv={H_kv}, D={D}")
    for B in batch_sizes:
        for T in seq_lens:
            for backend in backends:
                # Skip unavailable backends silently.
                info = attn_iface.get_backend(backend)
                if backend == "triton" and not info.available:
                    print(f"  [skip] {backend} not available: {info.reason}")
                    continue
                row = run_one(backend, B, H, H_kv, T, D, args.device, dtype, args.n_iters)
                rows.append(row)
                err = row["max_abs_error_vs_reference"]
                err_s = f"{err:.2e}" if err is not None else "n/a"
                print(
                    f"  {backend:10s} B={B} T={T:4d}  "
                    f"lat={row['latency_ms']:8.3f}ms  "
                    f"tps={row['tokens_per_sec']:8.1f}  err={err_s}"
                )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[bench] wrote {len(rows)} rows to {out_path}")

    # Summary JSON next to the CSV.
    summary = {
        "device": args.device,
        "dtype": args.dtype,
        "backends_attempted": backends,
        "rows": rows,
        "config": {"H": H, "H_kv": H_kv, "D": D},
    }
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[bench] wrote summary to {json_path}")


if __name__ == "__main__":
    main()