# W4 — notes.md

## The arithmetic-intensity story

GPU compute is cheap; GPU memory bandwidth is the bottleneck. A naive attention implementation does:

1. Materialize `(B, H, T, S)` logits in HBM — `O(T * S)` memory.
2. Softmax over S.
3. Multiply by V — read logits back.

For `T = S = 4096`, that's a `4096 × 4096` matrix per (B, H) — easily hundreds of MB. H100 has 80GB but `T = S = 8192` blows past L2 cache and you sit on HBM bandwidth for the rest of your life.

Flash-Attention (Dao et al. 2022/2023):

- Tile Q / K / V into SRAM-sized blocks.
- Never materialize the full `(T, S)` matrix.
- Online softmax: keep running `(m_i, l_i, acc)` statistics across blocks, single rescaling pass per block.

Memory complexity goes from `O(T²)` to `O(T)`. At `T = 4096, H = 32, fp16`, that's the difference between ~1 GB and ~16 MB.

## Triton vs cuDNN

- **cuDNN's Flash-Attention** is heavily tuned, lives at a lower level than Triton (uses CUTLASS + shared memory primitives). At small T and standard shapes, it wins.
- **Triton** is a Python-ish kernel language that compiles to PTX. We can write per-shape autotuning and capture-block choice that cuDNN doesn't expose.
- The interesting regime where Triton wins is large T (≥ 2K), large B, and shapes that aren't in cuDNN's standard library (e.g., sliding-window attention, custom mask patterns).

## The multi-GPU story (not yet implemented)

- **DDP** — replicate the model on each GPU, all-reduce gradients each step. Simple, but each GPU has the full model in memory.
- **FSDP / ZeRO-3** — shard model parameters + gradients + optimizer state across GPUs. Memory scales with `1 / world_size`. Useful when model doesn't fit on one GPU.
- **Tensor parallel** — split individual layers across GPUs (e.g., split attention heads). Lower communication volume per step, but harder to implement.
- **Sequence parallel** — split the sequence dimension across GPUs. Combined with TP for very long contexts.

For HW2, the brief is "at least one real multi-GPU experiment using DDP/FSDP/TP" — DDP is the easiest entry point.

## Reading

- `docs/reading/`
- `COURSE.md §2 W4`

## Open questions

- For the W4 multi-GPU benchmark, what's a sensible baseline? bf16 baseline-100m on 1 GPU vs 4 GPUs with DDP?
- Does scaling efficiency drop because of communication, because of small batch per GPU, or because the model is too small to amortize?