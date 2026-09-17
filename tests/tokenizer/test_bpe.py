"""Tokenizer smoke tests — runnable as a script (no pytest dependency)."""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))

from src.token_to_agent.tokenizer.bpe import (
    BPETokenizer, NUM_SPECIAL, BOS_ID, EOS_ID, PAD_ID, UNK_ID,
)


SAMPLE_TEXTS = [
    "hello world",
    "Hello, world!",
    "Once upon a time",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "1234567890",
    "你好，世界。",
    "mixing 中文 and English",
    "the quick brown fox jumps over the lazy dog",
]


def test_train_basic():
    tok = BPETokenizer.train(SAMPLE_TEXTS * 10, target_vocab_size=NUM_SPECIAL + 256 + 64)
    assert tok.stats_dict()["vocab_size"] >= NUM_SPECIAL + 256
    assert tok.stats_dict()["compression_bytes_per_token"] > 1.0
    print("[PASS] train_basic — vocab_size =", tok.stats_dict()["vocab_size"])


def test_encode_decode_roundtrip():
    tok = BPETokenizer.train(SAMPLE_TEXTS * 10, target_vocab_size=NUM_SPECIAL + 256 + 64)
    for s in SAMPLE_TEXTS:
        ids = tok.encode(s)
        back = tok.decode(ids)
        assert back == s, f"roundtrip failed: {s!r} -> {back!r}"
    print("[PASS] roundtrip — all sample texts decode identically")


def test_special_tokens():
    tok = BPETokenizer.train(SAMPLE_TEXTS * 5, target_vocab_size=NUM_SPECIAL + 256 + 16)
    ids = tok.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    # Special tokens are skipped during decode.
    assert tok.decode(ids) == "hello"
    print("[PASS] special_tokens — BOS/EOS handling correct")


def test_save_load():
    tok = BPETokenizer.train(SAMPLE_TEXTS * 10, target_vocab_size=NUM_SPECIAL + 256 + 64)
    tmp = "/tmp/_test_tokenizer.json"
    tok.save(tmp)
    tok2 = BPETokenizer.load(tmp)
    for s in SAMPLE_TEXTS:
        assert tok.encode(s) == tok2.encode(s)
    print("[PASS] save_load — encode output identical after round-trip")


def test_byte_level_coverage():
    """Every byte 0..255 must be representable."""
    tok = BPETokenizer.train(["hello"], target_vocab_size=NUM_SPECIAL + 256 + 4)
    # Try a 1-byte string for each possible byte value.
    for b in range(256):
        s = bytes([b]).decode("latin-1")
        ids = tok.encode(s)
        back = tok.decode(ids)
        assert back == s, f"byte {b} failed: {s!r} -> {back!r}"
    print("[PASS] byte_level_coverage — all 256 bytes round-trip")


if __name__ == "__main__":
    test_train_basic()
    test_encode_decode_roundtrip()
    test_special_tokens()
    test_save_load()
    test_byte_level_coverage()
    print("\nAll tokenizer tests passed.")