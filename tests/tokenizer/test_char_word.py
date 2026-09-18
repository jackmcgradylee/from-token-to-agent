"""W1 tests for CharTokenizer and WordTokenizer.

Run from repo root:
    PYTHONPATH=. .venv/bin/python tests/tokenizer/test_char_word.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.token_to_agent.from_scratch.tokenizer.char import (  # noqa: E402
    BOS_ID,
    CharTokenizer,
    EOS_ID,
    NUM_SPECIAL,
    PAD_ID,
    UNK_ID,
    VOCAB_SIZE,
)
from src.token_to_agent.from_scratch.tokenizer.word import WordTokenizer  # noqa: E402


# ----- tiny test harness -----------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, msg: str = "") -> None:
    _results.append((name, ok, msg))
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {name}{(' — ' + msg) if msg else ''}")


# ----- char tokenizer ------------------------------------------------

def test_char_vocab_size() -> None:
    ct = CharTokenizer()
    _check(
        "char.vocab_size == 260",
        ct.vocab_size == VOCAB_SIZE == 260,
        f"got {ct.vocab_size}",
    )


def test_char_special_tokens_first() -> None:
    ct = CharTokenizer()
    # IDs 0..3 are specials; first byte (0x00) maps to NUM_SPECIAL
    _check(
        "char.special tokens at fixed IDs 0..3",
        all(i in ct.id_to_byte for i in range(NUM_SPECIAL)) is False
        and ct.byte_to_id[0] == NUM_SPECIAL,
    )


def test_char_round_trip_english() -> None:
    ct = CharTokenizer()
    s = "Hello, world!"
    ids = ct.encode(s)
    back = ct.decode(ids)
    _check("char.round_trip(english)", back == s, f"{s!r} -> {back!r}")


def test_char_round_trip_cjk() -> None:
    s = "你好，世界。"
    ids = CharTokenizer().encode(s)
    back = CharTokenizer().decode(ids)
    _check("char.round_trip(cjk)", back == s, f"{s!r} -> {back!r}")


def test_char_round_trip_code() -> None:
    s = "def f(x): return x*2"
    ids = CharTokenizer().encode(s)
    back = CharTokenizer().decode(ids)
    _check("char.round_trip(code)", back == s, f"{s!r} -> {back!r}")


def test_char_round_trip_emoji() -> None:
    s = "Today 🎉 v1.0"
    ids = CharTokenizer().encode(s)
    back = CharTokenizer().decode(ids)
    _check("char.round_trip(emoji)", back == s, f"{s!r} -> {back!r}")


def test_char_byte_level_zero_unk() -> None:
    """Char/byte tokenizer has zero UNK by construction."""
    ct = CharTokenizer()
    # ANY UTF-8 string round-trips without UNK
    for s in ["你好", "🔭-heavy", "𝕳𝖊𝖑𝖑𝖔", "\x00\x01\x02", "ascii"]:
        ids = ct.encode(s)
        _check(
            f"char.no_unk({s!r})",
            UNK_ID not in ids,
            f"ids={ids}",
        )


def test_char_add_bos_eos() -> None:
    ids = CharTokenizer().encode("hi", add_bos=True, add_eos=True)
    _check(
        "char.bos/eos wrapping",
        ids[0] == BOS_ID and ids[-1] == EOS_ID,
        f"ids={ids}",
    )


def test_char_save_load() -> None:
    ct = CharTokenizer()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "char.json"
        ct.save(path)
        loaded = CharTokenizer.load(path)
        s = "你好 world 🎉"
        _check(
            "char.save_load.round_trip",
            loaded.encode(s) == ct.encode(s),
        )


def test_char_length_equals_bytes() -> None:
    """By construction, len(encode(s)) == len(s.encode('utf-8'))."""
    for s in ["hello", "你好", "def f():", "abc 🔥 xyz"]:
        ids = CharTokenizer().encode(s)
        _check(
            f"char.len==bytes({s!r})",
            len(ids) == len(s.encode("utf-8")),
            f"tokens={len(ids)} bytes={len(s.encode('utf-8'))}",
        )


# ----- word tokenizer ------------------------------------------------

_CORPUS = [
    "Hello, world!",
    "你好，世界。",
    "机器学习 是 人工智能 的 分支",
    "def greet(name: str): return name",
    "for (i = 0; i < n; i++) sum += a[i];",
    "今天 是个好日子",
    "Transformer uses self-attention",
]


def test_word_vocab_grows_with_corpus() -> None:
    """Word vocab is built from training data; size depends on distinct tokens."""
    wt = WordTokenizer().train(_CORPUS)
    _check(
        "word.vocab > 4 (specials only)",
        wt.vocab_size > NUM_SPECIAL,
        f"vocab={wt.vocab_size}",
    )


def test_word_round_trip_english() -> None:
    wt = WordTokenizer().train(_CORPUS)
    s = "Hello, world!"
    back = wt.decode(wt.encode(s))
    _check("word.round_trip(english)", back == s, f"{s!r} -> {back!r}")


def test_word_round_trip_cjk_with_space() -> None:
    """CJK with explicit spaces round-trips. CJK without spaces is split
    char-by-char (lossless) but produces different unit boundaries."""
    wt = WordTokenizer().train(_CORPUS)
    s = "今天 是个好日子"
    back = wt.decode(wt.encode(s))
    _check("word.round_trip(cjk spaced)", back == s, f"{s!r} -> {back!r}")


def test_word_cjk_unk_with_unseen_script() -> None:
    """A character never seen in training must map to <unk>."""
    wt = WordTokenizer().train(["hello world"])
    # Use a CJK char NOT in training (assuming train had only ASCII):
    s = "你好"
    ids = wt.encode(s)
    _check(
        "word.cjk unseen -> unk",
        all(i == UNK_ID for i in ids),
        f"ids={ids}",
    )


def test_word_code_round_trip() -> None:
    wt = WordTokenizer().train(_CORPUS)
    s = "def greet(name)"
    back = wt.decode(wt.encode(s))
    _check("word.round_trip(code)", back == s, f"{s!r} -> {back!r}")


def test_word_whitespace_preserved() -> None:
    """Whitespace tokens that are in the vocab must round-trip. Whitespace
    is preserved by pre_tokenization; if the specific whitespace run is
    unseen in training it falls back to <unk>. We use a whitespace run
    that appears in the training corpus."""
    wt = WordTokenizer().train(_CORPUS + ["   spaces   "])
    s = "   spaces   "
    back = wt.decode(wt.encode(s))
    _check("word.whitespace preserved", back == s, f"{s!r} -> {back!r}")


def test_word_save_load() -> None:
    wt = WordTokenizer().train(_CORPUS)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "word.json"
        wt.save(path)
        loaded = WordTokenizer.load(path)
        _check(
            "word.save_load.vocab_size",
            loaded.vocab_size == wt.vocab_size,
            f"{loaded.vocab_size} vs {wt.vocab_size}",
        )
        _check(
            "word.save_load.encode equal",
            loaded.encode("hello world") == wt.encode("hello world"),
        )


# ----- comparative check -------------------------------------------

def test_char_vs_word_seq_length_difference() -> None:
    """English prose: char sequence is roughly word-sequence * average word length.
    Use longer prose so single-letter words like 'a' and 'i' don't dominate."""
    wt = WordTokenizer().train(_CORPUS)
    s = (
        "The quick brown fox jumps over the lazy dog while philosophers "
        "argue about whether consciousness emerges from computation."
    )
    c_len = len(CharTokenizer().encode(s))
    w_len = len(wt.encode(s))
    # Prose-like English: chars / words ~ 5–7 (incl. spaces and punctuation)
    _check(
        "english char-seq is several times longer than word-seq",
        c_len > w_len * 2,
        f"char={c_len} word={w_len} (ratio={c_len/w_len:.1f}x)",
    )


# ----- main ----------------------------------------------------------

def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            _check(f"{t.__name__} (assertion)", False, str(e))
        except Exception as e:  # noqa: BLE001
            _check(f"{t.__name__} (exception)", False, f"{type(e).__name__}: {e}")

    failed = [r for r in _results if not r[1]]
    total = len(_results)
    print()
    print(f"=== {total - len(failed)} / {total} passed ===")
    if failed:
        print("FAILED:")
        for name, _, msg in failed:
            print(f"  {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())