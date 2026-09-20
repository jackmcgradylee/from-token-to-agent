---
extension_id: EXT-W1-01
title: BPE vocab size sweep on a small corpus
week: W1
stage: post-W1 (extension on top of HW1 Foundations)
status: complete
created: 2026-09-17
updated: 2026-09-17
tags: [tokenization, bpe, ablation, vocab-size, saturation]
hardware: CPU-only (no CUDA on the dev host, irrelevant to this sweep)
---

# EXT-W1-01 — BPE Vocab Size Sweep

> **TL;DR**: On a small toy corpus (~30 KB replicated 30×), BPE saturates at
> **actual_vocab = 530**. Asking for `target_vocab ≥ 1024` does **not** produce
> a bigger vocabulary — `min_pair_freq=2` filters every candidate pair, the
> merge loop stops, and the tokenizer plateaus. Best bytes/token = **1.214**,
> achieved at `target_vocab=512`; everything beyond is free entropy (more
> parameters, identical compression).

## 1. Problem

W1 already shipped a Char / Word / BPE comparison on `experiments/w01/exp-001`.
That experiment answered *which* tokenizer wins on a tiny corpus (BPE: zero
UNK, shorter sequences). It did not answer:

> **At what vocab size does BPE stop paying off on this corpus?**

The naive assumption — "bigger vocab = more merges = better compression" — is
clearly wrong in principle: once the corpus has been fully covered by merges,
additional merges have nothing left to add. But where is the cliff on
*this* corpus? And does the BPE trainer's `target_vocab_size` actually
matter past saturation, or does the loop just terminate early?

## 2. Motivation

Picking `vocab_size` for a production tokenizer is a real engineering
decision — GPT-2 picked 50,257, LLaMA-2 picked 32,000, Qwen-2 picked
152,064. Every choice trades off:

- **Embedding table size**: `(vocab × d_model)` — paid on every forward pass.
- **Sequence compression**: `bytes/token` — fewer tokens = shorter context.
- **UNK rate**: only matters if the tokenizer is closed-vocab; our BPE is
  byte-level so it never UNKs.

If `target_vocab > vocab_saturation`, you pay the embedding-table cost
without the compression benefit. This experiment exists to **measure** the
saturation cliff directly, on code I wrote myself, instead of taking it on
faith from the literature.

## 3. Method

- **Tokenizer**: the `BPETokenizer` from `src/token_to_agent/from_scratch/tokenizer/bpe.py`.
  Byte-level base (256 bytes + 4 special tokens = 260 minimum). Trained with
  `min_pair_freq=2`.
- **Corpus**: `experiments/w01/exp-001-char-word-bpe-comparison/samples.txt`
  (13 English/CJK/code/digit sentences, ~700 bytes original) replicated
  **30×** → ~21 KB of training text.
- **Eval set**: the same file's `## eval` section (5 held-out lines including
  rare CJK chars, emojis, and uncommon C++ operators). 272 bytes.
- **Sweep**: `target_vocab_size ∈ {260, 512, 1024, 2048, 4096}`. For each
  target we train a fresh BPE from scratch and encode the eval set.
- **Metrics**: `actual_vocab`, `num_merges`, `train_seconds`, `eval_bytes`,
  `eval_tokens`, `bytes_per_token` (= `eval_bytes / eval_tokens`), `eval_unk`.

A diagnostic fix was needed before this could run cleanly: the original
`_load_eval` skipped any line starting with `#`, which meant the section
marker `## eval` was also skipped, and the eval set ended up empty (the
bug that originally produced `bytes_per_token = 0.000`). Section markers
(`## train` / `## eval`) are now distinguished from single-`#` comments.

## 4. Experimental setup

- **Hardware**: CPU only (the dev host, no CUDA). This is a tokenizer sweep,
  not a model training run — CPU is the right tool.
- **Software**: Python 3.11 + `dataclasses`. No external deps.
- **Randomness**: deterministic — BPE merge order is fully determined by the
  input corpus and `min_pair_freq`. No seeds needed.
- **Single variable**: `target_vocab_size`. Everything else held fixed.

## 5. Results

| Target vocab | Actual vocab | Merges learned | Train (s) | Eval tokens | Bytes/token | UNK |
| --- | --- | --- | --- | --- | --- | --- |
| 260 | 260 | 0 | 0.015 | 272 | 1.000 | 0 |
| 512 | 512 | 252 | 0.149 | 224 | 1.214 | 0 |
| 1,024 | **530** | **270** | 0.153 | 224 | 1.214 | 0 |
| 2,048 | 530 | 270 | 0.148 | 224 | 1.214 | 0 |
| 4,096 | 530 | 270 | 0.149 | 224 | 1.214 | 0 |

Reproduce with:

```bash
PYTHONPATH=. .venv/bin/python extensions/w01/ext-w1-01-bpe-vocab-sweep/runner.py \
    --replicate 30 --targets 260,512,1024,2048,4096
```

Raw numbers in [`results/sweep.json`](./results/sweep.json). Auto-rendered
markdown table in [`results/sweep.md`](./results/sweep.md).

## 6. Analysis

**Where the saturation sits**: this corpus saturates at `actual_vocab = 530`.
At `target_vocab ≥ 1024` the trainer's merge loop runs out of pairs with
`freq ≥ 2` after **270 merges**, and the resulting vocabulary freezes at
530. Asking for `target_vocab = 4096` does not change a single downstream
number — same compression, same token IDs, just a bigger (and mostly empty)
vocab dictionary that the trainer hands back to you.

**Where the compression wins**: bytes/token improves **0.214** (1.000 → 1.214)
when going from byte-level (260) to the saturated BPE (530). All of that
win comes from collapsing frequent byte sequences into single tokens —
specifically the CJK bigrams (`你好`, `机器` etc.) and the common ASCII
subwords (`the`, `ing`, `tion`). Past 512 merges there is no corpus
material left to collapse.

**Why this matters for production**: GPT-2 (50,257) and LLaMA-2 (32,000)
are both **well below** their saturation points on the corpora they trained
on — they had material to keep merging but **deliberately stopped**, trading
compression for a smaller softmax in the LM head. The opposite error
(picking vocab *at* the saturation point) wastes embedding capacity.

**Limits of this experiment**:

1. **Corpus is toy-sized**. On FineWeb-Edu or Wikipedia, the saturation
   curve continues into the hundreds of thousands of tokens. This sweep
   measures *where the saturation is on a 21 KB blob* — not on a real
   distribution.
2. **No latency sweep**. Encoding speed depends on vocab size (larger `vocab`
   → bigger hash table to look up). Production choices should also factor
   in tokenize-time ms — that's a separate `EXT-W1-02`.
3. **No UNK to measure**. Byte-level BPE never UNKs. A closed-vocab BPE
   would show a non-zero UNK rate at low vocab sizes — that's a different
   question and would need a different tokenizer.

## 7. Reproduction

```bash
cd ~/project/llm-from-scratch-to-agent
.venv/bin/python extensions/w01/ext-w1-01-bpe-vocab-sweep/runner.py \
    --replicate 30 --targets 260,512,1024,2048,4096
```

Outputs:

- `results/sweep.json` — raw `VocabSweepRow` dicts.
- `results/sweep.md` — auto-rendered markdown table.

## 8. Extensions (next steps)

- **EXT-W1-02** — tokenize-time latency vs vocab size on the same corpus.
  Measures a different cost axis (compute) of the same trade-off.
- **EXT-W1-03** — repeat the sweep on a real corpus (e.g. a 1 GB slice of
  FineWeb-Edu). Expect saturation to shift from 530 into the hundreds of
  thousands.
- **EXT-W1-04** — closed-vocab BPE (with UNK). Track UNK rate vs vocab
  size on a held-out OOV-heavy eval set.

## See also

- [`experiments/w01/exp-001-char-word-bpe-comparison/`](../exp-001/) —
  parent W1 experiment this extension builds on.
- [`src/token_to_agent/from_scratch/tokenizer/bpe.py`](../../../../src/token_to_agent/from_scratch/tokenizer/bpe.py) —
  the BPE implementation being swept.
- [`docs/signal-ladder.md`](../../../../docs/signal-ladder.md) — W1
  textbook deliverable (Tokenization is step 1 of the ladder).