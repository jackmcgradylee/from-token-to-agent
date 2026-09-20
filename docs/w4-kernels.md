# W4 — Compute, Kernels, Parallelism (textbook deliverable)

This document is the W4 reading material: the *why* behind the
kernels. It pairs with the source code in
`src/token_to_agent/from_scratch/kernels/` and the experiments in
`experiments/w04/`.

## 1. Where does the time go in a Transformer?

For a single forward pass at sequence length T, batch B, H heads,
head dim D:

| Stage | FLOPs | Naive memory | Tiled (Flash) memory |
|---|---:|---:|---:|
| Q @ K^T | 2 · B · H · T² · D | O(T² · H · B) | O(T · D · H · B) in SRAM |
| softmax | ~5 · B · H · T² | O(T² · H · B) | O(T · BLOCK_M · H · B) in SRAM |
| A @ V | 2 · B · H · T² · D | O(T · D · H · B) | O(T · D · H · B) in SRAM |
| MLP (×2) | 4 · B · T · d_model · d_ff | O(B · T · d_ff) | same |

Attention is the **only** stage where the intermediate is O(T²).
Everything else is linear in T. So at long contexts (T ≥ 4k), the
attention intermediate dominates memory traffic and the FLOPs-per-
byte ratio (arithmetic intensity, AI) collapses.

## 2. Arithmetic intensity — the metric that explains everything

AI = FLOPs / bytes-touched. Hardware has a single number for "how
many FLOPs per byte the chip can sustain":

| Hardware | Peak FP16 AI (FLOP/byte) |
|---|---:|
| A100 | ~1500 |
| H100 | ~2500 |
| the dev host (CPU only) | irrelevant; ~1 |

If your kernel's AI < hardware peak, you're **bandwidth-bound** and
the kernel is memory-limited; if AI > peak, you're **compute-bound**
and the kernel is FLOP-limited.

Our estimator (`flops_estimator.py`) at (T=4096, D=64, fp16):

| Path | AI (FLOP/byte) |
|---|---:|
| MLP (large matmul) | ~100 |
| Naive attention (with (T, S) intermediate) | ~120 |
| Flash-style tiled attention | ~2050 |

Naive attention has the *same* AI as the MLP (≈ 100). The win from
tiling is not "more AI" — it's that flash attention's bytes-touched
shrinks dramatically (no (T, S) round-trip to HBM), so the *time*
shrinks even though the FLOPs are the same.

## 3. FlashAttention-2 — what the kernel does

We implement the Flash-2 forward in
`kernels/triton_attention.py`. The core loop is:

```
for each query block M_i (size BLOCK_M × D):
    load Q_i (stays in SRAM)
    m = -inf, l = 0, acc = 0
    for each key block N_j:
        load K_j, V_j
        qk = Q_i @ K_j^T * scale
        qk = causal_mask(qk, M_i, N_j)
        m_new = max(m, rowmax(qk))
        alpha = exp(m - m_new)
        p = exp(qk - m_new)
        l = l * alpha + rowsum(p)
        acc = acc * alpha + p @ V
        m = m_new
    out[i] = acc / l
```

This is *online softmax* (Milakov & Gimelshein 2018): the running
max and denominator carry the softmax statistics across tiles, so we
never need to know the global max before writing the final output.
The math is algebraically equivalent to `softmax(Q @ K^T) @ V` but
the (T, S) intermediate never leaves SRAM.

## 4. Why GQA is a memory win, not a compute win

In grouped-query attention, `H_q > H_kv`. Each Q head reads from
exactly one KV head (via `n_rep = H_q / H_kv`). The Triton kernel
parameterises this:

```python
h_kv_idx = h_idx // n_rep
```

In the inner loop, each query block still does the same number of
FLOPs as MHA — but K/V cache memory is **smaller by `n_rep`**. So
GQA doesn't speed up compute, it shrinks memory pressure (and the
KV cache).

## 5. Backward — the 4-line derivation

For the reference path, the backward is just the chain rule applied
to `O = softmax(Q K^T scale) V`:

```
D      = (dO * O).sum(-1)                          # (B, H, T, 1)
dP     = dO @ V^T                                  # (B, H, T, S)
dS     = (dP - D * P) * scale                      # (B, H, T, S) where P = softmax
dQ     = dS @ K                                    # (B, H, T, D)
dK     = dS^T @ Q                                  # (B, H_kv, S, D)
dV     = P^T @ dO                                  # (B, H_kv, S, D)
```

If GQA is active, `dK`/`dV` come out at the expanded `H` shape; we
reshape-and-sum across the `n_rep` Q heads that shared each KV head.

For the Triton path, the backward is harder: Dao et al. 2023
("FlashAttention-2") re-derives a tile-by-tile backward that doesn't
need to store the (T, S) matrix. The skeleton is in
`kernels/triton_attention_backward.py` (future work; gated on CUDA).

## 6. Distributed training — DDP

W4 doesn't ask for a re-implementation of DDP, but we should
understand what it does:

```
for each step:
  1. Each replica i (rank 0..N-1) holds a full set of params θ_i = θ
  2. Each replica processes its shard of the batch: x_i, y_i ~ batch_i
  3. Each replica computes local gradient: g_i = ∇L(f(x_i; θ_i), y_i)
  4. All-reduce: g_avg = mean(g_0, ..., g_{N-1})
  5. Each replica updates: θ_i = θ_i - lr * g_avg
  6. (optional) Broadcast updated θ to all replicas
```

Steps 4 and 5 are why your effective batch size is `batch_per_shard × N`,
not `batch_per_shard × N × something_else`. The communication cost
(step 4) is dominated by the gradient size; on NCCL over NVLink this
is roughly 1-10 μs per all-reduce, comparable to one matmul forward.

The toy correctness test in `systems/distributed/ddp_reference.py`
verifies the gradient-mean claim on a CPU host by running two local
forwards and averaging their gradients.

## 7. What FSDP adds

FSDP goes one step further: not just the batch is sharded, but the
**parameters themselves** are sharded. Each replica only holds
`1/N` of each parameter tensor at any moment; the others are
fetched on-demand (all-gather) before the forward / backward pass
through that layer.

FSDP memory savings: roughly `1/N` of the model parameters per GPU,
plus optimiser state (Adam: 2× params for moments → 2× saving too).
This is what lets you train 70B-class models on 8× A100-40GB where
DDP would OOM.

`systems/distributed/fsdp_smoke.py` is a launchable wrapper; running
it on a CUDA host with `torchrun --nproc_per_node=N` gives you a
minimal end-to-end demonstration.

## 8. What we measure in `exp-001` / `exp-002`

| Metric | Definition |
|---|---|
| `forward_ms` | Median of 5 timed runs, after 3 warmup iterations. |
| `fwd_bwd_ms` | Same, but includes `loss.backward()` after `out.sum()`. |
| `peak_mem_mb` | Process-level RSS delta on CPU; on CUDA this would be `torch.cuda.max_memory_allocated`. |
| `max_abs_err` | Max absolute difference vs the `reference` forward output (fp32). |
| `tflops` | Effective FLOPs achieved by the forward pass: `4 · B · H · T² · D / forward_ms / 1e9`. |

The seq-len sweep (`exp-002`) plots forward latency vs T on a
log-log plot. On a CUDA host, the `triton` row should have a
visibly flatter slope than the `naive` row at large T — that's the
"Flash beats naive" story in one chart.

## 9. Why the harness is useful even on CPU

The relative ordering of the backends is real even when running on
CPU with Triton falling back to reference. The `pytorch` (SDPA)
row should beat `reference` and `naive` because PyTorch's SDPA
fuses the softmax with the matmul; on CUDA, the `triton` row joins
the top with another factor of speedup.

The harness doesn't claim to give absolute performance numbers on a
machine that doesn't have the relevant hardware. It does claim to
produce CSV / plot artifacts that are *correct* (the CSV row for a
given (B, H, T, D) on a CPU host today should match the same row on
a CUDA host with `cuda=False` set, modulo the Triton row being the
real one).

## 10. Open TODOs gated on CUDA

- Triton backward kernel (we have the reference backward; the
  tile-by-tile Triton version is the next step).
- Real FSDP launch with `torchrun --nproc_per_node=N` (skeleton in
  `systems/distributed/fsdp_smoke.py`).
- Real FlashAttention vs SDPA benchmark on H100 / A100 (the data
  in `exp-001` would update to show the real Flash speedup).
- KV-cache benchmarking (W5 territory but the kernel would benefit).

These are scoped but deferred — see
`experiments/w04/README-DEV-HOST-LIMITATIONS.md`.