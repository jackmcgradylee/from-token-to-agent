# W2 — notes.md

## 2017 Transformer vs 2026 decoder-only LLM

What survives from Vaswani et al. (2017)?

- Q @ K^T / softmax / @V — yes, still the same
- Residual connections — yes, but **pre-norm** (LayerNorm/RMSNorm before the sublayer), not post-norm
- Sinusoidal position encoding — **gone**. Replaced by RoPE or similar relative-position encodings.
- Encoder + decoder blocks — **gone**. Decoder-only with causal masking.
- Tied input/output embeddings — common in small models, often untied in frontier models.

What was added:

- **RMSNorm** (Zhang & Sennrich 2019) — LayerNorm without the mean-centering step. Cheaper, similar stability.
- **SwiGLU** (Shazeer 2020) — gating FFN, replaces GeLU/ReLU MLP. LLaMA-style.
- **GQA** (Ainslie et al. 2023) — KV head sharing. Reduces KV memory without quality loss.
- **RoPE** (Su et al. 2021) — rotary positional embeddings.
- **KV cache** — incremental decode is now mandatory, not a benchmark trick.
- **Tied embeddings** — common in smaller models.

What frontier models add on top of the baseline (MoE, MTP, DSA) lives in EXT, not the baseline.

## Why this matters for the course

- W2 baseline = exactly what the toy model implements. No abstractions hiding the math.
- HW2 systems (W4) swaps the attention path; that swap is only meaningful if you understand the path you're replacing.
- HW4 post-training (W8+) starts from a model of this shape (100M scratch or 1.5B–9B open base).

## Open questions

- Where does `head_dim` come from? In baseline-100m: `d_model=768, n_heads=12 → head_dim=64`. Could also be configured independently (`head_dim != d_model / n_heads`), which is what some recent models do.
- For GQA: `n_kv_heads=1` is MQA, `n_kv_heads=n_heads` is MHA, in between is GQA. What ratio gives best quality / memory tradeoff?