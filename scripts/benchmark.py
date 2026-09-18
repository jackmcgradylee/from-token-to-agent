"""CLI entry: benchmark attention / kernels / serving (used by W4, W5).

Usage (W4 — attention backends):
    PYTHONPATH=src python scripts/benchmark.py attention \
        --device cpu --dtype float32 \
        --backends reference,pytorch \
        --seq-lens 64,128,256 --batch-sizes 1,4 \
        --output experiments/w04/exp-001-triton-vs-pytorch-vs-reference/benchmark-cpu.csv

Usage (W5 — serving economics, TODO):
    PYTHONPATH=src python scripts/benchmark.py serving --help
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
for p in (str(_repo_root), str(_repo_root / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from src.token_to_agent.from_scratch.kernels import interface as attn_iface
from src.token_to_agent.from_scratch.kernels import reference_attention


def cmd_attention(args) -> int:
    import torch as _torch

    dtype = {"float32": _torch.float32, "float16": _torch.float16, "bfloat16": _torch.bfloat16}[args.dtype]
    backends = [b.strip() for b in args.backends.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if args.device == "cuda" and not _torch.cuda.is_available():
        print("[bench] CUDA not available; falling back to CPU")
        args.device = "cpu"

    H, H_kv, D = 12, 4, 64  # baseline-100m attention math layer

    rows = []
    print(f"[bench] device={args.device} dtype={args.dtype} backends={backends}")
    print(f"[bench] shape: H={H}, H_kv={H_kv}, D={D}")
    for B in batch_sizes:
        for T in seq_lens:
            for backend in backends:
                info = attn_iface.get_backend(backend)
                if backend == "triton" and not info.available:
                    print(f"  [skip] {backend} not available: {info.reason}")
                    continue

                _torch.manual_seed(0)
                q = _torch.randn(B, H, T, D, device=args.device, dtype=dtype) * 0.5
                k = _torch.randn(B, H_kv, T, D, device=args.device, dtype=dtype) * 0.5
                v = _torch.randn(B, H_kv, T, D, device=args.device, dtype=dtype) * 0.5
                n_rep = H // H_kv

                def fn():
                    attn_iface.call(backend, q, k, v, is_causal=True, n_rep=n_rep, training=False)

                if B * T * T < 8 * 1024 * 1024:
                    ref = reference_attention.reference_attention_forward(
                        q.float(), k.float(), v.float(), is_causal=True, n_rep=n_rep,
                    )
                    try:
                        out = attn_iface.call(backend, q, k, v, is_causal=True, n_rep=n_rep, training=False)
                        err = (out.float() - ref).abs().max().item()
                    except Exception:
                        err = float("nan")
                else:
                    err = None

                samples = []
                for _ in range(2):
                    fn()
                for _ in range(args.n_iters):
                    gc.collect()
                    t0 = time.perf_counter()
                    fn()
                    samples.append((time.perf_counter() - t0) * 1000)
                samples.sort()
                median_ms = samples[len(samples) // 2]
                tokens = B * T
                tokens_per_sec = tokens / (median_ms / 1000)

                rows.append({
                    "backend": backend,
                    "B": B, "H": H, "H_kv": H_kv, "T": T, "D": D,
                    "device": args.device, "dtype": args.dtype,
                    "latency_ms": round(median_ms, 3),
                    "tokens_per_sec": round(tokens_per_sec, 1),
                    "max_abs_error_vs_reference": err,
                })
                err_s = f"{err:.2e}" if err is not None else "n/a"
                print(f"  {backend:10s} B={B} T={T:4d}  lat={median_ms:8.3f}ms  tps={tokens_per_sec:8.1f}  err={err_s}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"device": args.device, "dtype": args.dtype, "backends_attempted": backends, "rows": rows}, f, indent=2)
    print(f"[bench] wrote {len(rows)} rows to {out_path}")
    return 0


def cmd_serving(args) -> int:
    print("[bench] serving mode not yet implemented (W5 deliverable).")
    print("        see weeks/w05-economics-of-inference/README.md for the plan.")
    return 1


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_attn = sub.add_parser("attention", help="Benchmark attention forward across backends")
    p_attn.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p_attn.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p_attn.add_argument("--backends", default="reference,pytorch,triton")
    p_attn.add_argument("--n-iters", type=int, default=5)
    p_attn.add_argument("--seq-lens", default="128,256,512,1024")
    p_attn.add_argument("--batch-sizes", default="1,4")
    p_attn.add_argument("--output", required=True)
    p_attn.set_defaults(func=cmd_attention)

    p_srv = sub.add_parser("serving", help="Benchmark serving economics (W5 — TODO)")
    p_srv.add_argument("--url", default=None)
    p_srv.add_argument("--output", default=None)
    p_srv.set_defaults(func=cmd_serving)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())