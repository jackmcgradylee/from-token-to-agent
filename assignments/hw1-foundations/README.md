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
| **W3** Training Dynamics & Scaling Laws | Toy baseline + scaling pilot (queued for next round) | 🔜 |

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
| **Total** | **59** | All pass on Jetson Nano (CPU-only). |

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

### 🔜 W3 — training dynamics + scaling-law hold-out (next)

The hold-out structure for HW3 is sketched in `experiments/w03/exp-001-toy-baseline/`
(only smoke baseline exists today). Following rounds will add the
scaling-law pilots at 20M / 50M / 100M, then the W4 Triton kernel pass.

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

Total wall time on Jetson Nano CPU: < 30 s.

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