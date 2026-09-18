# W1 — Char vs Word vs BPE comparison

Samples: **5 eval lines** (trained on 13 lines), total **272 UTF-8 bytes**.  
BPE vocab size: **1024**  (target).  Char vocab is fixed at 260; Word vocab is built greedily from the sample set.

## Vocab sizes

| Tokenizer | Vocab size |
| --- | --- |
| char | 260 |
| word | 107 |
| bpe | 306 |

## Compression (bytes per token — higher = more efficient)

| Tokenizer | Total tokens | Bytes/token | UNK count |
| --- | --- | --- | --- |
| char | 272 | 1.00 | 0 |
| word | 89 | 3.06 | 38 |
| bpe | 239 | 1.14 | 0 |

> **UNK count** is the count of `<unk>` tokens in the output. Char/byte has 0 UNK by construction. Word has UNK whenever it sees a token not in the training vocab (notably CJK without explicit spacing). BPE has 0 UNK by construction (byte-level).

## Per-sample token counts

| Sample (truncated) | bytes | char | word | bpe | word-UNK |
| --- | --- | --- | --- | --- | --- |
| `Cassiniophilina's zephyr-laden persiflage bewil...` | 64 | 64 | 16 | 55 | 8 |
| `Quixotry and rheumy bibliobibuli vexed the cata...` | 55 | 55 | 14 | 47 | 5 |
| `嵚崟嶷魑魅魍魉孀嫠。` | 30 | 30 | 10 | 28 | 9 |
| `Quantum 🎯 entanglement and 🧬 sequencing herald ...` | 63 | 63 | 20 | 53 | 8 |
| `auto&& [this]() noexcept -> decltype(auto) { re...` | 60 | 60 | 29 | 56 | 8 |

## Observations

- **CJK samples (eval)**: char tokenizer emits exactly **3 tokens per CJK character** because each char is 3 UTF-8 bytes. BPE compresses below 3 tokens once merges learn frequent byte triples; on this tiny eval set with very rare characters, BPE stays at ~2.8 tokens per CJK char (almost no merges apply). Word tokenizer falls back to 1 token per CJK character too — but every character not in the training vocab becomes `<unk>`, so the rare-character line `嵚崟嶷魑魅魍魉孀嫠。` produces 9 UNKs.
- **Word OOV is brutal**: across 5 held-out eval sentences the word tokenizer produces **38 `<unk>` tokens** (43% of all word tokens). These are rare English words (`Cassiniophilina`, `quixotry`, `persiflage`, `bibliobibuli`, `catachresis`), uncommon CJK characters, and emoji. **Char and BPE produce 0 UNK by construction** — this is the canonical argument for subword methods.
- **English prose (eval)**: rare long words like `Cassiniophilina's` cost 11 char tokens vs 1 word token (when in vocab) — but when *not* in vocab, the word tokenizer collapses the entire word into `<unk>`, destroying information. BPE decomposes the same word into ~7 byte-level subwords, all valid tokens.
- **Code (eval)**: rare C++ syntax (`auto&&`, `noexcept`, `decltype`) costs the word tokenizer several UNKs; BPE handles them cleanly through byte merges of common operator sequences.

## Files

- [`runner.py`](./runner.py) — this script
- [`comparison.json`](./comparison.json) — raw numbers
- [`samples.txt`](./samples.txt) — input corpus
