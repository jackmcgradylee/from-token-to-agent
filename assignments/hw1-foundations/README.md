# HW1 — Foundations

> **Status:** 🟢 **complete** (W1 + W2 + EXT-W1-01 shipped; 59/59 tests pass).
>
> Tagged: [`v0.1-hw1-foundations`](../../) — the HW1 baseline is frozen.
> All later assignments (HW2 / HW3 / HW4) build on top of it, not on a copy.

## Scope

This assignment answers: *can we hand-build a small language model, from
tokenizer to checkpoint, without leaning on HuggingFace Trainer or any
pre-trained checkpoint?* Everything in HW1 lives in
`src/token_to_agent/from_scratch/`.

## Weeks covered

| Week | Topic | Status |
| --- | --- | --- |
| **W1** Three Paradigm Shifts | Char / Word / BPE tokenizer comparison + signal-ladder write-up | ✅ |
| **W2** Architecture Revisited | RMSNorm vs LayerNorm + MHA vs GQA ablations | ✅ |
| **W3** Training Dynamics & Scaling Laws | 5-point scaling pilot + power-law fit + hold-out prediction | ✅ |

This README is updated incrementally as each week lands; until W3 closes,
HW1 status is **"W1 + W2 done; W3 in progress"**.

## Implementation (`src/token_to_agent/from_scratch/`)

### Tokenizer (`tokenizer/`)

- `bpe.py` — byte-level BPE, byte base (256 + 4 specials = 260 minimum), `train`/`encode`/`decode`/`save`/`load`, `vocab_size` property.
- `char.py` — UTF-8 byte-level char tokenizer, vocab = 260, **0 UNK by construction**.
- `word.py` — mixed-language word tokenizer (whitespace + CJK char-by-char + digits + UNK fallback). Closed-vocab; the only one of the three that produces `<unk>`.

### Model (`model/`)

- `model.py` — full `Transformer` (RMSNorm by default, GQA by default, RoPE, SwiGLU).
- `attention.py` — multi-head attention with RoPE, optional GQA via `n_kv_heads`.
- `layernorm.py` — vanilla LayerNorm, paired with `rmsnorm.py` for ablation.
- `rmsnorm.py` — RMSNorm (the default in `TransformerConfig.norm_kind="rmsnorm"`).
- `rope.py` — rotary position embeddings.
- `swiglu.py` — SwiGLU MLP.
- `config.py` — `TransformerConfig` dataclass with `norm_kind` and `n_kv_heads` switches.

### Tests (`tests/`)

| Test file | Assertions | What it locks in |
| --- | --- | --- |
| `tests/tokenizer/test_bpe.py` | 5 | BPE byte-level round-trip, special tokens, vocab property. |
| `tests/tokenizer/test_char_word.py` | 26 | Char/word round-trip, OOV behaviour, comparative metrics. |
| `tests/model/test_transformer.py` | 4 | Param-count breakdown, forward shape, RoPE effect. |
| `tests/model/test_ablation.py` | 19 | RMSNorm vs LayerNorm params + GQA KV-cache shrinkage. |
| `tests/kernels/test_attention.py` | 5 | Attention shape / KV cache / numerical sanity (CPU no-CUDA branch). |
| **Total** | **59** | All pass on the dev host (CPU-only). |

## Experiments

### ✅ W1 — exp-001-char-word-bpe-comparison

Train/eval split on a 13-sentence mixed-language corpus with 5 held-out
lines (rare CJK, emojis, uncommon C++ operators).

| Tokenizer | Vocab | UNK on eval | Bytes / token |
| --- | --- | --- | --- |
| Char (byte-level) | 260 | **0** | 1.00 |
| Word (mixed-lang) | 107 | **38 (43%)** | 3.06 |
| BPE (target 1024) | 306 | **0** | 1.14 |

Key finding: word tokenizers collapse rare tokens to `<unk>`, losing all
signal; byte-level methods never do.

Full numbers: [`experiments/w01/exp-001-char-word-bpe-comparison/comparison.md`](../../experiments/w01/exp-001-char-word-bpe-comparison/comparison.md).

### ✅ W2 — exp-001 RMSNorm vs LayerNorm

Same `TransformerConfig` switched via `norm_kind`:

| Variant | Total params | Δ |
| --- | --- | --- |
| LayerNorm | `d_total + 640` | +0 |
| RMSNorm | `d_total` | **−640** |

Δ comes from `(2 × n_layers + 1) × d_model = (2·2 + 1) × 128 = 640` bias
vectors. Loss difference < 0.003 nats — under CPU noise.

### ✅ W2 — exp-002 MHA vs GQA

| Variant | n_kv_heads | Total params | KV cache / layer (B=4, T=512) | Final loss |
| --- | --- | --- | --- | --- |
| MHA | 4 | 557,696 | 2,097,152 | 6.972 |
| GQA-2 | 2 | 524,928 | 1,048,576 | 6.974 |
| GQA-1 (= MQA) | 1 | 508,544 | **524,288** | 6.990 |

KV cache shrinks by exactly `n_kv_heads / n_heads` (verified in tests).
Loss within 0.018 nats across the range.

### ✅ EXT-W1-01 — BPE vocab size sweep (post-W1 extension)

| Target vocab | Actual vocab | Merges learned | Bytes/token | UNK |
| --- | --- | --- | --- | --- |
| 260 | 260 | 0 | 1.000 | 0 |
| 512 | 512 | 252 | **1.214** | 0 |
| 1,024 | **530** | 270 | 1.214 | 0 |
| 2,048 | 530 | 270 | 1.214 | 0 |
| 4,096 | 530 | 270 | 1.214 | 0 |

On this 21 KB toy corpus, BPE saturates at `actual_vocab = 530`. Asking
for `target ≥ 1024` produces no additional merges — the trainer's
`min_pair_freq=2` loop terminates early.

Full write-up: [`extensions/w01/ext-w1-01-bpe-vocab-sweep/README.md`](../../extensions/w01/ext-w1-01-bpe-vocab-sweep/README.md).

### ✅ W3 — scaling pilot + power-law hold-out prediction

Five iso-data scaling pilots with `d_model × n_layers` swept log-spaced
across one and a half orders of magnitude in N:

| Pilot | Target N | Actual N | d_model | n_layers | val_loss @ step 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1m | 1M | 1,642,496 | 128 | 8 | 4.1495 |
| 2_5m | 2.5M | 3,590,400 | 160 | 12 | 3.7409 |
| 5m | 5M | 6,887,616 | 192 | 16 | (see results) |
| 10m | 10M | 15,280,384 | 256 | 20 | (see results) |
| 20m | 20M | 33,359,680 | 320 | 28 | (see results) |

All five share: same 1 MB toy corpus (`data/raw/sample.txt`), same
byte-level BPE vocab (560 merges), same lr schedule (3e-4 peak →
cosine to 3e-5 with 20-step warmup), same AdamW (β=0.9/0.95), same
effective batch 8, same seed=42, 100 training steps. The sweep is
**iso-data** (D fixed), not iso-compute — a deliberate simplification
that lets us attribute loss differences to capacity alone.

The hold-out test (`experiments/w03/exp-007-scaling-law-fit/`) fits
`L(N) = L_inf + a · N^(-α)` on {1M, 2.5M, 5M, 10M} by 1-D grid search
over `L_inf` with closed-form OLS for `a, α`, then predicts the 20M
point. The relative prediction error is the honest measurement of
whether the W3 sweep actually behaves like a power law.

Implementation:

- Shared harness: [`experiments/w03/_common.py`](../../experiments/w03/_common.py) — one `run_scaling_pilot(config, output_dir)` that every pilot's `runner.py` calls.
- Five configs: [`configs/hw1/scaling-{1m,2_5m,5m,10m,20m}.yaml`](../../configs/hw1/).
- Corpus generator: [`scripts/make_sample_corpus.py`](../../scripts/make_sample_corpus.py) — deterministic 1 MB TinyStories-style text (seed=42).
- Five pilot folders with their own `runner.py`, `loss_curve.csv`, `metadata.json`, `samples.txt`, `loss_curve.png` under [`experiments/w03/`](../../experiments/w03/).

See [`experiments/w03/exp-007-scaling-law-fit/results/fit.md`](../../experiments/w03/exp-007-scaling-law-fit/results/fit.md) for the canonical fit numbers and hold-out error. Textbook-style write-up: [`docs/scaling-law.md`](../../docs/scaling-law.md).

## Reproduction

A fresh environment needs only two commands to validate HW1:

```bash
# 1. All unit tests
for f in tests/{tokenizer,model}/test_*.py tests/kernels/test_*.py; do
    PYTHONPATH=. .venv/bin/python "$f" || exit 1
done

# 2. The two W2 ablations
PYTHONPATH=. .venv/bin/python experiments/w02/exp-001-layernorm-vs-rmsnorm/runner.py
PYTHONPATH=. .venv/bin/python experiments/w02/exp-002-mha-vs-gqa/runner.py

# 3. The EXT-W1-01 sweep
PYTHONPATH=. .venv/bin/python extensions/w01/ext-w1-01-bpe-vocab-sweep/runner.py \
    --replicate 30 --targets 260,512,1024,2048,4096
```

Total wall time on the dev host CPU: < 30 s.

## Pass criteria (from `COURSE.md` §6 HW1)

- [x] BPE training and encode/decode work without external tokenizer-training libraries.
- [x] Char + Word tokenizers exist and pass tests.
- [x] RMSNorm vs LayerNorm ablation produces a numeric result with parameter breakdown.
- [x] MHA vs GQA ablation produces a numeric result with KV-cache breakdown.
- [x] `tests/` are green on a fresh checkout.
- [x] Reproduction is documented and one-shot.

## See also

- `weeks/w01-paradigm-shifts/README.md` — W1 week-level journal.
- `weeks/w02-architecture-revisited/README.md` — W2 week-level journal.
- `docs/signal-ladder.md` — W1 textbook deliverable.
- `report.md` — long-form HW1 report (filled in after W3 closes).