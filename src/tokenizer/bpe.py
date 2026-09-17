"""Byte-level BPE tokenizer, written from scratch.

Why from scratch:
  - HW1 DoD requires a tokenizer I trained.
  - HuggingFace tokenizers is great, but a hand-written ~200-line BPE makes
    the algorithm transparent and lets us extend it later (Unigram, BBPE, etc.).

Algorithm:
  1. Pre-tokenize text into a stream of UTF-8 bytes (0..255) per word, with
     whitespace as its own "word" so spaces round-trip through encode/decode.
  2. Build an initial vocab of 256 byte tokens + special tokens.
  3. Iteratively merge the most-frequent adjacent pair, adding one new token
     each round, until target vocab size is reached.
  4. Save: vocab (token bytes -> id), merges (list of (a, b) pairs in order).

Encoding text = apply merges in rank order to each word.
Decoding = lookup id -> bytes, join, utf-8 decode.

Reference: Sennrich et al. 2016; byte-level BPE from GPT-2 (Radford et al. 2019).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# --- Special token IDs (fixed) ----------------------------------------------------

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
NUM_SPECIAL = len(SPECIAL_TOKENS)


@dataclass
class TokenizerStats:
    vocab_size: int = 0
    num_merges: int = 0
    corpus_bytes: int = 0
    corpus_tokens: int = 0
    compression_ratio: float = 0.0  # bytes per token
    training_seconds: float = 0.0


@dataclass
class BPETokenizer:
    """Byte-level BPE tokenizer.

    Attributes:
        vocab: dict[bytes, int]  (token bytes -> id)
        id_to_bytes: list[bytes] (id -> token bytes); index 0..vocab_size-1
        merges: list[tuple[int, int]]  ordered list of (token_id_a, token_id_b)
            merges[i] is the (a, b) that produced token id (NUM_SPECIAL + 256 + i)
        bpe_ranks: dict[(int, int), int]  (a, b) -> rank (lower = apply first)
    """

    vocab: dict[bytes, int] = field(default_factory=dict)
    id_to_bytes: list[bytes] = field(default_factory=list)
    merges: list[tuple[int, int]] = field(default_factory=list)
    bpe_ranks: dict[tuple[int, int], int] = field(default_factory=dict)

    # ------------------------------------------------------------------ training

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        target_vocab_size: int,
        min_pair_freq: int = 2,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Train a byte-level BPE on an iterable of texts.

        target_vocab_size must be >= NUM_SPECIAL + 256 (i.e., at least the
        base byte vocab). Each merge adds 1 to the vocab.
        """
        if target_vocab_size < NUM_SPECIAL + 256:
            raise ValueError(
                f"target_vocab_size must be >= {NUM_SPECIAL + 256} "
                f"(special + base bytes); got {target_vocab_size}"
            )

        tok = cls()

        # Init vocab: special tokens, then 256 byte tokens.
        for s in SPECIAL_TOKENS:
            tok.vocab[s.encode("utf-8")] = len(tok.id_to_bytes)
            tok.id_to_bytes.append(s.encode("utf-8"))
        for b in range(256):
            tok.vocab[bytes([b])] = len(tok.id_to_bytes)
            tok.id_to_bytes.append(bytes([b]))

        # Convert all text into per-word sequences of byte-token ids.
        corpus_bytes = 0
        word_freq: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            for word in _split_on_whitespace(text):
                if not word:
                    continue
                word_bytes = word.encode("utf-8")
                ids = tuple(word_bytes)
                word_freq[ids] += 1
                corpus_bytes += len(word_bytes)

        # Map raw byte ids to actual vocab ids (offset by NUM_SPECIAL).
        byte_id_offset = NUM_SPECIAL
        word_freq_actual: Counter[tuple[int, ...]] = Counter()
        for ids, freq in word_freq.items():
            actual_ids = tuple(b + byte_id_offset for b in ids)
            word_freq_actual[actual_ids] = freq

        num_target_merges = target_vocab_size - len(tok.id_to_bytes)
        t0 = time.time()
        for merge_step in range(num_target_merges):
            # Compute pair frequencies across the whole corpus (weighted by word freq).
            pair_freq: Counter[tuple[int, int]] = Counter()
            for ids, freq in word_freq_actual.items():
                for a, b in zip(ids, ids[1:]):
                    pair_freq[(a, b)] += freq

            if not pair_freq:
                if verbose:
                    print(f"[bpe] No more pairs at step {merge_step}; stopping early.")
                break

            best_pair, best_count = pair_freq.most_common(1)[0]
            if best_count < min_pair_freq:
                if verbose:
                    print(f"[bpe] Best pair freq {best_count} < {min_pair_freq}; stopping.")
                break

            # Add new merged token to vocab.
            a_bytes = tok.id_to_bytes[best_pair[0]]
            b_bytes = tok.id_to_bytes[best_pair[1]]
            new_bytes = a_bytes + b_bytes
            if new_bytes in tok.vocab:
                continue
            new_id = len(tok.id_to_bytes)
            tok.vocab[new_bytes] = new_id
            tok.id_to_bytes.append(new_bytes)
            tok.merges.append(best_pair)
            tok.bpe_ranks[best_pair] = len(tok.merges) - 1

            # Apply merge to all words (single pass over word_freq_actual).
            new_word_freq: Counter[tuple[int, ...]] = Counter()
            for ids, freq in word_freq_actual.items():
                new_ids = _apply_merge(ids, best_pair, new_id)
                new_word_freq[new_ids] = freq
            word_freq_actual = new_word_freq

            if verbose and (merge_step + 1) % 500 == 0:
                print(
                    f"[bpe] step {merge_step + 1}/{num_target_merges} "
                    f"merged {a_bytes!r}+{b_bytes!r} (freq={best_count})"
                )

        elapsed = time.time() - t0

        total_tokens = sum(len(ids) * freq for ids, freq in word_freq_actual.items())
        tok._stats = TokenizerStats(  # type: ignore[attr-defined]
            vocab_size=len(tok.id_to_bytes),
            num_merges=len(tok.merges),
            corpus_bytes=corpus_bytes,
            corpus_tokens=total_tokens,
            compression_ratio=corpus_bytes / max(total_tokens, 1),
            training_seconds=elapsed,
        )
        return tok

    # ------------------------------------------------------------------ encode

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(BOS_ID)
        for word in _split_on_whitespace(text):
            if not word:
                continue
            byte_ids = [b + NUM_SPECIAL for b in word.encode("utf-8")]
            # Apply BPE merges in rank order.
            ids.extend(_apply_bpe(byte_ids, self.bpe_ranks))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    # ------------------------------------------------------------------ decode

    def decode(self, ids: Iterable[int]) -> str:
        pieces = []
        for i in ids:
            if i < NUM_SPECIAL:
                # Special tokens decode to nothing.
                continue
            pieces.append(self.id_to_bytes[i])
        raw = b"".join(pieces)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ I/O

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "id_to_bytes": [b.hex() for b in self.id_to_bytes],
            "merges": [list(m) for m in self.merges],
            "stats": getattr(self, "_stats", None).__dict__
            if hasattr(self, "_stats")
            else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls()
        tok.id_to_bytes = [bytes.fromhex(h) for h in payload["id_to_bytes"]]
        tok.vocab = {b: i for i, b in enumerate(tok.id_to_bytes)}
        tok.merges = [(int(a), int(b)) for a, b in payload["merges"]]
        tok.bpe_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        if payload.get("stats"):
            from .bpe import TokenizerStats
            tok._stats = TokenizerStats(**payload["stats"])  # type: ignore[attr-defined]
        return tok

    def stats_dict(self) -> dict:
        if not hasattr(self, "_stats"):
            return {}
        s = self._stats  # type: ignore[attr-defined]
        return {
            "vocab_size": s.vocab_size,
            "num_merges": s.num_merges,
            "corpus_bytes": s.corpus_bytes,
            "corpus_tokens": s.corpus_tokens,
            "compression_bytes_per_token": round(s.compression_ratio, 4),
            "training_seconds": round(s.training_seconds, 2),
        }


# --- helpers --------------------------------------------------------------------


def _split_on_whitespace(text: str) -> list[str]:
    """Split text into tokens where whitespace is its own token.

    "hello world  foo" -> ["hello", " ", "world", " ", " ", "foo"]
    Whitespace round-trips cleanly through encode/decode.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            out.append(text[i:j])
            i = j
        else:
            j = i
            while j < n and not text[j].isspace():
                j += 1
            out.append(text[i:j])
            i = j
    return out


def _apply_merge(
    ids: tuple[int, ...], pair: tuple[int, int], new_id: int
) -> tuple[int, ...]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    a, b = pair
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


def _get_pairs(ids: list[int]) -> set[tuple[int, int]]:
    return {(ids[i], ids[i + 1]) for i in range(len(ids) - 1)}


def _apply_bpe(
    ids: list[int], ranks: dict[tuple[int, int], int]
) -> list[int]:
    """Apply BPE merges in rank order until no more merges apply.

    Reference algorithm (GPT-2 BPE):
        pairs = get_pairs(word)
        while True:
            bigram = min(pairs, key=lambda pair: ranks.get(pair, inf))
            if bigram not in ranks: break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                j = find index of (first, second) starting at i
                append word[i:j]
                append (first+second) token  -- but we need its id, not bytes
                ...
    Trick: we work with **ids** throughout, and the merged id is the FIRST
    token id in the pair (we update that id in place to the new vocab id).
    This works because BPE always creates a new token; the pair's two ids are
    contiguous, and we replace both with the new id.
    """
    if len(ids) < 2:
        return list(ids)

    # Build pair -> positions mapping
    parts = list(ids)  # mutable copy; entries will be replaced with new ids
    while True:
        pairs = {(parts[i], parts[i + 1]): i for i in range(len(parts) - 1)}
        if not pairs:
            break
        # Find the lowest-rank pair.
        best_pair = min(pairs, key=lambda p: ranks.get(p, float("inf")))
        if best_pair not in ranks:
            break
        first, second = best_pair
        # Replace all non-overlapping occurrences.
        new_parts: list[int] = []
        i = 0
        n = len(parts)
        # We need the NEW token id; it's not in ranks directly. The new id is
        # NOT known here, but we don't actually need it: BPE merges always
        # produce the *next* vocab id (vocab_size_after = NUM_SPECIAL+256+rank+1).
        # Compute it from rank: new_id = NUM_SPECIAL + 256 + ranks[best_pair].
        # However this requires knowing the token order, which is implicit in
        # the ranks dict. We just compute it inline.
        new_id = NUM_SPECIAL + 256 + ranks[best_pair]
        while i < n:
            if i + 1 < n and parts[i] == first and parts[i + 1] == second:
                new_parts.append(new_id)
                i += 2
            else:
                new_parts.append(parts[i])
                i += 1
        parts = new_parts
        if len(parts) < 2:
            break
    return parts