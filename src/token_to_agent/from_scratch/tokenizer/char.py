"""Byte-level char tokenizer (W1).

The simplest possible tokenizer: each UTF-8 byte is its own token.
Vocabulary is fixed at 256 byte tokens + 4 special tokens = 260 total.

Why byte-level (not Unicode codepoint):
  - Zero UNK for any input. Any UTF-8 string decodes losslessly.
  - Trivially comparable to BPE: BPE's 256-byte base IS char-level
    before any merges happen. So char-length is an upper bound on
    the token count for any byte-level subword tokenizer.
  - CJK characters cost 3 tokens each (UTF-8), demonstrating the
    'CJK inflation' problem that motivates BPE's merges.

This is the textbook baseline. It is NOT useful for production
language modeling — sequence lengths explode — but it isolates the
tokenizer from vocabulary effects and is the cleanest unit of
comparison.

Reference: Radford et al. 2019 (GPT-2 byte-level BPE), where the
base vocabulary is exactly these 256 bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


# --- Special token IDs (fixed, must match the other tokenizers) ----
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
NUM_SPECIAL = len(SPECIAL_TOKENS)

VOCAB_SIZE = NUM_SPECIAL + 256  # 260, fixed


class CharTokenizer:
    """Byte-level tokenizer: text -> bytes -> ids, and back."""

    name = "char"

    def __init__(self) -> None:
        # byte -> id (special tokens first, then 0..255)
        self.byte_to_id: dict[int, int] = {
            b: NUM_SPECIAL + b for b in range(256)
        }
        self.id_to_byte: dict[int, int] = {
            i: b for b, i in self.byte_to_id.items()
        }

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    def train(self, texts: Iterable[str]) -> "CharTokenizer":
        """Char tokenizer has no training — vocab is fixed by definition.

        Accepts (and ignores) texts so all three tokenizers share the
        same `train(...)` interface.
        """
        return self

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.byte_to_id[b] for b in text.encode("utf-8")]
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        raw = bytearray()
        for i in ids:
            i = int(i)
            if i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            if i == UNK_ID:
                continue
            b = self.id_to_byte.get(i)
            if b is None:
                # out-of-vocab id — substitute with U+FFFD
                continue
            raw.append(b)
        return raw.decode("utf-8", errors="replace")

    # --- I/O ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Char tokenizer is fully determined by its name; persist a
        marker file so .load() can verify the right class is being
        restored.
        """
        import json
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump({"kind": self.name, "vocab_size": self.vocab_size}, f)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        import json
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("kind") != cls.name:
            raise ValueError(f"file {p} is not a char tokenizer: {meta}")
        return cls()