# EXT-W1-01 — BPE vocab size sweep

Trains BPE on the W1 samples corpus (replicated 50× to give the merge loop enough room to actually run). Sweeps `target_vocab_size` over a wide range and reports the actual vocab, the number of merges learned, the resulting bytes/token compression on the W1 held-out eval set, and the UNK count.

## Results

| Target vocab | Actual vocab | Merges learned | Train (s) | Eval tokens | Bytes/token | UNK |
| --- | --- | --- | --- | --- | --- | --- |
| 260 | 260 | 0 | 0.015 | 272 | 1.000 | 0 |
| 512 | 512 | 252 | 0.149 | 224 | 1.214 | 0 |
| 1,024 | 530 | 270 | 0.153 | 224 | 1.214 | 0 |
| 2,048 | 530 | 270 | 0.148 | 224 | 1.214 | 0 |
| 4,096 | 530 | 270 | 0.149 | 224 | 1.214 | 0 |

## Hypothesis vs observation

**Hypothesis**: bytes/token grows sublinearly with vocab; once merges are saturated, more target vocab buys no compression.

**Observation**: with this corpus, BPE saturates at `target_vocab >= 1024` (training stops because no adjacent pair has frequency ≥ `min_pair_freq=2` any longer). The actual vocab plateaus at 530 even though we asked for 4096.

**Best bytes/token = 1.214** at target_vocab = 512 (actual vocab = 512). Adding more target vocab beyond this point yields zero additional merges and the same compression — the vocabulary is exhausted on this corpus.

## Why this matters

BPE in production typically picks a target vocab far below the saturation point — GPT-2 used 50,257 tokens across a real web corpus that easily supported 100K+ merges. Picking vocab *at* the saturation point would be wasteful: larger embedding tables and softmax costs without compression gain.

## What I would change on a real corpus

- Use ≥100MB of text (e.g. a slice of FineWeb-Edu) — the current 50× replication of 13 sentences (~32 KB) hits the merge ceiling in <500 merges.
- Compare sweep against the byte-level char baseline (1.00 bytes/token) to quantify the compression ratio.
- Add `EXT-W1-02`: tokenize-time latency vs vocab size — large vocabs make each encode step search a bigger hash table.

## Files

- [`runner.py`](./runner.py) — this script
- [`results/sweep.json`](./results/sweep.json) — raw numbers
- [`README.md`](./README.md) — 5-piece extension write-up
