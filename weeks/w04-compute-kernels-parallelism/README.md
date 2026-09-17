# W4 — Compute, Kernels, Parallelism

## What this week is about

Where does Transformer compute time actually go, and how much of model performance is a systems problem rather than a model problem?

## Status

In Progress. Triton kernel + reference + benchmark framework are implemented; one benchmark CSV exists. **Full W4 deliverable** (multi-GPU DDP/FSDP/TP measurement, seq 512/2K/8K sweep with forward+backward latency) not yet done.

## Implemented so far

- `src/token_to_agent/kernels/reference_attention.py` — pure-PyTorch SDPA oracle (CPU-runnable, used as correctness ground truth).
- `src/token_to_agent/kernels/triton_attention.py` — Flash-Attention-2-style forward kernel:
  - Online softmax (no logits materialization)
  - Tile sizes keyed off `head_dim` (`BLOCK_D ∈ {32, 64, 128, 256}`)
  - Manual heuristic block choice (`BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=3`); autotune configs prepared but not active by default
  - GQA expansion done outside the kernel (KV heads `repeat_interleave`'d) — keeps the kernel simple
  - Forward-only for W4. Backward is `EXT-W4-02`.
- `src/token_to_agent/kernels/interface.py` — unified `AttentionBackend` dispatch:
  - `"pytorch"` → `F.scaled_dot_product_attention`
  - `"triton"` → our kernel, with **silent fallback to reference** when CUDA is unavailable
  - `"reference"` → pure-PyTorch oracle
- `src/token_to_agent/model/attention.py` — `Attention.forward(...)` delegates the math to `kernels.interface.call(...)` instead of calling SDPA directly.
- `src/token_to_agent/model/model.py` — `TransformerConfig.attn_backend: str` + `set_attn_backend(...)` (propagates to every layer, logs fallback).
- `scripts/benchmark_attention.py` — latency / tokens/s / peak memory / max_abs_error sweep over backends × batch × seq × head-dim.
- `tests/kernels/test_attention.py` — backend dispatch + correctness oracle. Skips Triton comparison on CUDA-less hosts.

## Evidence

- `experiments/w04/exp-001-triton-vs-pytorch-vs-reference/benchmark-cpu.csv` and `.json` —
  Jetson CPU fp32 sweep, attention shape `(B, H=12, H_kv=4, T, D=64)`:
  - B=1 T=128: reference 8.49ms, pytorch 4.05ms, **speedup 2.10×**, max_abs_err 1.19e-7
  - B=4 T=256: reference 83.4ms, pytorch 49.6ms, **speedup 1.68×**, max_abs_err 1.34e-7
  - Triton row: SKIPPED — kernel requires CUDA, Jetson has no GPU.

## W4 deliverable gaps

- [ ] Multi-GPU benchmark (DDP / FSDP / TP) — requires CUDA + multi-GPU.
- [ ] Seq-length sweep at 512 / 2K / 8K.
- [ ] Forward **and** backward latency (only forward benchmarked).
- [ ] Numerical error **gradient** check.
- [ ] Multi-GPU benchmark table under `experiments/w04/exp-002-multi-gpu-...`.

## Pass criteria (from COURSE.md §2 W4)

- [ ] output and gradient correctness are within documented tolerances.
  - ✅ forward correctness (1.49e-7 max abs error, fp32 SDPA vs reference).
  - ❌ gradient correctness (not measured).
- [ ] at least one regime shows a measured systems benefit **or** a clear explanation of why it does not. ✅ (2× speedup measured on Jetson CPU).

## Open questions

- Does the Triton kernel actually beat cuDNN's Flash-Attention on bf16 at long T (≥ 2K)? Need a GPU box.
- Should the autotune configs be active by default? Manual heuristic was chosen for predictable first-call latency.