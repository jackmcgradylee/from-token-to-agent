# W3 — notes.md

## The training stack

```
raw text
↓
tokenizer (src/token_to_agent/tokenizer/)
↓
tokenized dataset (TokenDataset)
↓
TransformerLM.forward(x, targets=y)
↓
loss = cross_entropy(logits, targets)
↓
optimizer.step() — AdamW + cosine LR
↓
checkpoint
↓
generation / eval
```

## Things that look small but break training

- **Weight decay** must apply only to Linear / Embedding weights, **not** to biases or norm scales. If you apply wd=0.1 to RMSNorm's `weight`, training diverges.
- **Gradient clipping** at 1.0 is mandatory in practice for the 0.1B scale; without it, loss spikes from rare token clusters.
- **`grad_accum > 1`** must divide `max_steps` evenly, otherwise the loop ends mid-accumulation and the optimizer doesn't step.
- **fp32 RMS inside RMSNorm** — running it in bf16 produces NaN within 100 steps.
- **Offset arithmetic in KV cache** — easy to off-by-one.

## Scaling-law intuition

For transformer LMs, Kaplan et al. (2020) and Chinchilla (Hoffmann et al. 2022) showed:

- `loss(N, D) ≈ E + A / N^α + B / D^β` where `N` is params, `D` is tokens.
- At small scale (`N ≤ 100M, D ≤ 1B`), the curve is noisy and depends heavily on tokenizer, data quality, and init.
- The interesting W3 pilot question is **not** whether the curve fits — at 20M/50M/100M it will — but **where it deviates from the textbook form**, because that's where the implementation choices matter.

## Reading

- `docs/reading/` (to be populated)
- `COURSE.md §2 W3`

## Open questions

- For the W3 pilot, should we hold data fixed and vary model, or vary both? The course brief says W3 is **model-side** scaling (data fixed); HW3 (W6–W7) is the data × model 2D scaling.
- What loss target is "converged enough" for the pilot? Textbook 0.1B models on TinyStories reach ~2.0 val loss at ~5B tokens. We don't have 5B tokens on the dev host.