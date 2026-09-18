"""Mixed-language word tokenizer (W1).

Tokenization rules (language-aware heuristic, no ML):
  - ASCII letter run  -> 1 token  ("hello" / "TransformerLM")
  - ASCII digit run   -> 1 token  ("12345")
  - ASCII whitespace  -> 1 token per run (preserved for round-trip)
  - ASCII punctuation -> 1 token each (",", ".", "(", etc.)
  - CJK char           -> 1 token each ("你好" -> ["你", "好"])
  - Other Unicode      -> 1 token per codepoint, or <unk> if not seen in train
  - Anything not in vocab after training -> <unk>

Vocab is built greedily from the training corpus: every distinct
token form observed becomes a vocab entry.

Why this design:
  - "Word" in English is whitespace+punc split.
  - CJK has no whitespace. The principled choice is char-per-CJK —
    this matches jieba's behavior on out-of-vocab Chinese, and lets
    us see the "word tokenizer on Chinese" pathology in the
    comparison report (mostly single-char vocab + UNK for names).
  - The OOV / UNK rate IS the killer stat for word tokenizers and
    shows why subword methods exist.

References:
  - Classical word tokenization (Manning et al., Stanford NLP).
  - Mixed-script handling: Heineman et al. 2016 (tweet tokenizer).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List


# --- Special token IDs (fixed, must match the other tokenizers) ----
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
NUM_SPECIAL = len(SPECIAL_TOKENS)

# Unicode category ranges (exposed for readability in tokenize()).
# CJK Unified Ideographs extension A + base + zero-width joiners + symbols.
_CJK_RE = re.compile(
    r"[\u3000-\u303f"        # CJK symbols / punctuation
    r"\u3400-\u4dbf"          # CJK ext A
    r"\u4e00-\u9fff"          # CJK Unified Ideographs
    r"\uf900-\ufaff"          # CJK compatibility
    r"\U00020000-\U0002a6df"  # CJK ext B
    r"\U0002a700-\U0002b73f"  # CJK ext C
    r"\U0002b740-\U0002b81f"  # CJK ext D
    r"]"
)


def _is_cjk(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


class WordTokenizer:
    """Whitespace + punctuation word tokenizer with CJK fallback."""

    name = "word"

    def __init__(self) -> None:
        # token-string -> id, special tokens first
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: dict[int, str] = {i: t for t, i in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # --- Pre-tokenization ------------------------------------------
    def pre_tokenize(self, text: str) -> List[str]:
        """Split text into a flat list of word / char / punc tokens.

        Rules:
          - Whitespace run          -> 1 token (preserved for round-trip)
          - ASCII letter/digit run  -> 1 token
          - ASCII punctuation       -> 1 token each
          - CJK char                -> 1 token each
          - Other Unicode codepoint -> 1 token each
        """
        tokens: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            cp = ord(ch)

            if ch.isspace():
                # capture full whitespace run
                j = i
                while j < n and text[j].isspace():
                    j += 1
                tokens.append(text[i:j])
                i = j
            elif _is_cjk(ch):
                tokens.append(ch)
                i += 1
            elif 0x21 <= cp <= 0x7E:
                # ASCII printable: letters/digits run vs single punc
                if ch.isalnum():
                    j = i
                    while j < n and text[j].isalnum() and ord(text[j]) <= 0x7E:
                        j += 1
                    tokens.append(text[i:j])
                    i = j
                else:
                    # punctuation / symbol -> 1 char
                    tokens.append(ch)
                    i += 1
            elif cp > 0x7E:
                # non-ASCII, non-CJK (e.g. emoji, Cyrillic, Arabic)
                tokens.append(ch)
                i += 1
            else:
                # control char: skip
                i += 1
        return tokens

    # --- Training --------------------------------------------------
    def train(self, texts: Iterable[str], min_freq: int = 1) -> "WordTokenizer":
        """Build vocabulary from pre-tokenized corpus."""
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(self.pre_tokenize(t))
        for tok, freq in counter.items():
            if freq < min_freq:
                continue
            if tok in self.token_to_id:
                continue
            new_id = len(self.token_to_id)
            self.token_to_id[tok] = new_id
            self.id_to_token[new_id] = tok
        return self

    # --- Encode / decode -------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        for tok in self.pre_tokenize(text):
            ids.append(self.token_to_id.get(tok, UNK_ID))
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        out: List[str] = []
        for i in ids:
            i = int(i)
            if i in (PAD_ID, BOS_ID, EOS_ID, UNK_ID):
                continue
            tok = self.id_to_token.get(i)
            if tok is None:
                continue
            out.append(tok)
        # Concatenation works because:
        #   - whitespace tokens already contain the spaces
        #   - punctuation tokens are single chars
        #   - CJK chars concatenate without separators
        return "".join(out)

    # --- I/O ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import json
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump({
                "kind": self.name,
                "vocab_size": self.vocab_size,
                "token_to_id": self.token_to_id,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        import json
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("kind") != cls.name:
            raise ValueError(f"file {p} is not a word tokenizer: {meta}")
        tok = cls()
        tok.token_to_id = {k: int(v) for k, v in meta["token_to_id"].items()}
        tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}
        return tok