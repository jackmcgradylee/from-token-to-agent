# W4 — Compute, Kernels, Parallelism

## What this week is about

Two questions:

1. **Where does the time go?** The attention matmul is O(T²) in sequence
   length. Naive implementations materialise the full (T, S) logit
   matrix in HBM; flash-style kernels keep it in SRAM and never
   write it back. The arithmetic-intensity gap between the two paths
   is what makes FlashAttention 5-10× faster on a GPU.
2. **Why does multi-GPU work at all?** We use PyTorch DDP /
   FSDP / Megatron (we don't reimplement them — see COURSE.md §九),
   and understand what they actually do: replicate the model on N
   GPUs, split the batch, all-reduce the gradients, average.

W4 doesn't ask for a 0.1B model — it asks for **kernel mechanics** and
**distributed mechanics** that you understand because you wrote (or
at least wrapped) them.

## Status

**Done on the CPU-runnable parts.** Triton forward kernel is complete
and self-tests against the reference. Backward is implemented as a
`torch.autograd.Function` for the reference path; the Triton backward
path is a stub (skeleton file) waiting for a GPU host. Benchmark
harness and seq-len sweep both run on CPU (with Triton falling back to
reference) and produce CSV/PNG outputs.

**CUDA-required parts deferred.** Real FlashAttention-style speedups,
FSDP launches, NCCL all-reduce — all gated behind CUDA. See
`experiments/w04/README-DEV-HOST-LIMITATIONS.md`.

## What was implemented

### Triton flash-attn-2 forward kernel
File: `src/token_to_agent/from_scratch/kernels/triton_attention.py`.

- One program = one (block of M queries, one (B, H)).
- Tiling in SRAM: Q is loaded once into registers, K/V blocks are
  loaded tile-by-tile.
- Online softmax: `m_i`, `l_i`, accumulator kept across K-block
  iterations with the standard Flash-2 update rule.
- GQA expansion: each Q head reads from `h_idx // n_rep`'s KV head.
- Causal masking: early-terminate K iteration when K block is fully
  above the diagonal.
- dtype-agnostic (fp32 / fp16 / bf16 via constexpr).

When CUDA is unavailable the module imports `triton = None` and the
kernel is skipped; the interface silently falls back to reference.

### Backward as `torch.autograd.Function`
File: `src/token_to_agent/from_scratch/kernels/reference_attention.py`.

- `ReferenceAttention` is a custom `torch.autograd.Function` whose
  forward saves (Q, K, V, weights) and whose backward computes
  (dQ, dK, dV) via the 4-line standard derivation.
- GQA is handled by summing dK/dV across the Q-heads that share each
  KV head.
- `naive_attention_forward` is a pure-PyTorch matmul path used as
  the *cross-check* in unit tests (gradients must match
  `ReferenceAttention`'s gradients to < 1e-4 relative error).

### Four-backend interface
File: `src/token_to_agent/from_scratch/kernels/interface.py`.

| Backend | Path | When |
|---|---|---|
| `pytorch` | `F.scaled_dot_product_attention` | Default (HW1). |
| `triton`  | `triton_attention_forward` | On CUDA; falls back silently otherwise. |
| `naive`   | `naive_attention_forward` | CPU-runnable baseline; used as the "naive" row in W4 bench. |
| `reference` | `reference_attention_forward` | Oracle for correctness tests. |

`list_backends()` returns all four with availability and reason.

### DDP/FSDP wrappers
File: `src/token_to_agent/systems/distributed/ddp_reference.py`.

- `setup_distributed()`: reads RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
  from env; auto-picks MASTER_PORT via `find_free_port()`. Single-process
  fallback returns (0, 1, 0) without calling `init_process_group`.
- `manual_all_reduce_sum/avg`: toy pure-Python all-reduce for
  testing the *primitive* without torch.distributed.
- `smoke_ddp_correctness`: a CPU-runnable test that verifies the
  mathematical claim "average gradients over N replicas" gives the
  same parameter update as a single forward on the concat batch.

### FLOPs / memory estimators
File: `src/token_to_agent/systems/profiling/flops_estimator.py`.

- `attention_flops(B, H, H_kv, T, S, D, causal)` — analytic FLOPs.
- `attention_mem(..., flash=True/False)` — peak memory with/without
  the (T, S) logit intermediate.
- `transformer_block_flops(...)` — full block FLOPs.
- `attention_mem` saves 32 KB on T=128 fp16 vs naive; the
  arithmetic-intensity test reports `flash is 17× naive` at T=4096.

### Benchmark + sweep
Files:
- `experiments/w04/exp-001-attn-bench/runner.py`
- `experiments/w04/exp-002-attn-seqlen/runner.py`

Both:
- run on CPU (Triton falls back),
- emit CSV + JSON + Markdown + (where matplotlib is available) PNG,
- measure median forward latency and forward+backward latency,
- compute TFLOPS achieved using the analytic FLOP count.

`exp-002` sweeps T ∈ {128, 256, 512, 1024, 2048} to show the O(T²)
latency growth on the four backends.

## Reproduction

```bash
# All on CPU.
PYTHONPATH=. .venv/bin/python tests/kernels/test_attention.py
PYTHONPATH=. .venv/bin/python src/token_to_agent/systems/distributed/ddp_reference.py
PYTHONPATH=. .venv/bin/python src/token_to_agent/systems/profiling/flops_estimator.py
PYTHONPATH=. .venv/bin/python experiments/w04/exp-001-attn-bench/runner.py
PYTHONPATH=. .venv/bin/python experiments/w04/exp-002-attn-seqlen/runner.py
```

On a CUDA host, run additionally:

```bash
# 4-GPU FSDP smoke (CUDA only)
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
    src/token_to_agent/systems/distributed/fsdp_smoke.py
```

## What I learned

- **Naive attention is O(T²) in both FLOPs and memory; Flash is O(T²) in
  FLOPs but O(T) in memory.** The arithmetic-intensity jump (17× at
  T=4096 in our estimator) is what makes FlashAttention the
  default backend on every modern training run.
- **GQA changes nothing about the math, just the memory layout**:
  the Q heads that share a KV head get the same K/V tile loaded once
  and reused. The Triton kernel factors this out via the
  `N_REP = H / H_KV` parameter.
- **Backward is mostly a re-derivation, not new math**: the custom
  `autograd.Function` is ~30 lines and matches the naive matmul
  path's gradients to 1e-4 relative error.
- **DDP averaging is "two forward passes, one update"**: writing a
  smoke test that mimics two replicas and averages gradients is a
  very effective way to internalise what distributed training
  actually does.
- **The CPU harness is useful even without CUDA**: the relative
  ranking of backends (`pytorch` SDPA > `reference` > `triton=ref
  fallback`) is a real signal. The absolute latencies are not — they
  invert once you have a GPU.

## Pass criteria (from `COURSE.md` §九 W4)

- [x] Triton Attention Kernel implemented (forward path; backward is
      `autograd.Function` on reference, Triton backward skeleton
      gated on CUDA).
- [x] Naive PyTorch / SDPA / Triton comparison via shared interface.
- [x] Sequence length sweep {128, 256, 512, 1024, 2048} producing
      latency/memory/throughput numbers.
- [x] Multi-GPU wrapper code present (DDP + FSDP smoke scripts) —
      execution gated on CUDA host.
- [ ] **CUDA-host multi-GPU run** (deferred to `experiments/w04/
      README-DEV-HOST-LIMITATIONS.md`).

## See also

- [`docs/w4-kernels.md`](../../docs/w4-kernels.md) — W4 textbook deliverable.
- [`experiments/w04/exp-001-attn-bench/results/bench.md`](../../experiments/w04/exp-001-attn-bench/results/bench.md)
- [`experiments/w04/exp-002-attn-seqlen/results/seqlen.md`](../../experiments/w04/exp-002-attn-seqlen/results/seqlen.md)
- [`experiments/w04/README-DEV-HOST-LIMITATIONS.md`](../../experiments/w04/README-DEV-HOST-LIMITATIONS.md)
- [`assignments/hw2-systems/README.md`](../../assignments/hw2-systems/README.md) — HW2 views W4.