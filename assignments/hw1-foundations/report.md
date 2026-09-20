# HW1 — Foundations · Report

> **Status: In Progress** — smoke verified only. Not yet tagged.

## 1. Problem

Build a tokenizer + Transformer LM from scratch and train an approximately 0.1B LM end-to-end on raw text. Implement, don't just instantiate.

## 2. Motivation

HW2 (systems) swaps the attention path; HW3 (data) retrains the tokenizer; HW4 (post-training) branches off the base checkpoint; Final (agents) consumes the model. None of those are meaningful if HW1 was just `from transformers import ...`.

## 3. Method

See the three week folders:
- `weeks/w01-paradigm-shifts/README.md`
- `weeks/w02-architecture-revisited/README.md`
- `weeks/w03-training-dynamics-scaling-laws/README.md`

## 4. Experimental setup

- **Tokenizer:** byte-level BPE, vocab size 8192 (target; toy run hit 559 due to 1 MB corpus size).
- **Model:** baseline-100m — d_model=768, n_layers=12, n_heads=12, n_kv_heads=4 (GQA), d_ff=2048 (SwiGLU), RoPE, RMSNorm, ~95M params. Toy config: d_model=256, 4 layers, ~2.5M params.
- **Training:** AdamW (β=0.9/0.95, eps=1e-8, wd=0.1), grad clip 1.0, cosine LR (3e-4 peak, 10% min), warmup, grad accum.

## 5. Results (smoke only)

| Metric | Value (toy-5m) |
|---|---|
| Params | 2,504,704 |
| Train tokens | 384,011 |
| Val tokens | 2,000 |
| Vocab | 559 |
| Compression | 2.72 bytes/token |
| Train loss | 6.43 → 2.27 |
| Val loss | 2.26 |
| Tokens/sec | 329 (dev CPU) |
| Peak memory | 1.4 GB |
| Wall time | 292.6 s |

Full 0.1B and 20M/50M/100M scaling pilot pending (W3 deliverable).

## 6. Analysis (preliminary)

- RMSNorm + SwiGLU + GQA + RoPE stack trains stably from step 0 (no spikes, smooth warmup→cosine descent).
- Toy 2.5M model at 200 steps learns local syntax and corpus vocabulary but cannot sustain long-range coherence (expected).
- fp32 RMS inside RMSNorm matters — without it, NaN within ~100 steps.

## 7. Reproduction

See `assignments/hw1-foundations/README.md` § Reproduction.

## 8. Extensions

- `EXT-W1-01` Byte-level BPE
- `EXT-W1-02` Unigram tokenizer
- `EXT-W1-03` Vocabulary-size ablation
- `EXT-W2-01` Tiny MoE
- `EXT-W2-02` Sparse attention
- `EXT-W2-03` Multi-token prediction head