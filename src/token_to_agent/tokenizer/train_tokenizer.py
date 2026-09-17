"""Tokenizer train/test driver.

Trains a BPE on a text corpus, prints stats, then round-trips a few sample
sentences through encode/decode to verify.

Usage:
    python -m src.tokenizer.train_tokenizer \\
        --input data/raw/sample.txt \\
        --output checkpoints/hw1/baseline-100m/tokenizer.json \\
        --vocab-size 2048 \\
        --sample-out assignments/hw1-foundations/results/tokenizer_samples.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bpe import BPETokenizer


SAMPLE_SENTENCES = [
    "Once upon a time, there was a little girl named Lily.",
    "The quick brown fox jumps over the lazy dog.",
    "Hello world! This is a test.",
    "In a small village by the sea,",
    "Tom and his friend went to the park.",
    "1234567890",
    "你好，世界。",  # Chinese bytes round-trip
    "Mixing 中文 and English: hello, world.",
    "   leading and trailing whitespace   ",
    "Repeated words: the the the the the.",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to a UTF-8 text file.")
    p.add_argument("--output", required=True, help="Where to save tokenizer.json.")
    p.add_argument(
        "--vocab-size",
        type=int,
        default=2048,
        help="Target vocab size (must be >= 260; default 2048 for toy runs).",
    )
    p.add_argument("--min-pair-freq", type=int, default=2)
    p.add_argument("--sample-out", default=None,
                   help="If given, write encode/decode samples to this file.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    texts = input_path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Filter empty lines but keep them as paragraph breaks during training.
    # For simplicity: train on non-empty lines.
    texts = [t for t in texts if t.strip()]

    print(f"[tok] training on {len(texts)} lines from {input_path}")
    tok = BPETokenizer.train(
        texts,
        target_vocab_size=args.vocab_size,
        min_pair_freq=args.min_pair_freq,
        verbose=args.verbose,
    )
    stats = tok.stats_dict()
    print("[tok] stats:", json.dumps(stats, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(out_path)
    print(f"[tok] saved to {out_path}")

    if args.sample_out:
        sample_path = Path(args.sample_out)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("# Tokenizer samples")
        lines.append(
            f"# vocab_size={stats['vocab_size']}, "
            f"compression={stats['compression_bytes_per_token']} bytes/token"
        )
        lines.append("")
        for s in SAMPLE_SENTENCES:
            ids = tok.encode(s)
            back = tok.decode(ids)
            lines.append(f"INPUT  : {s!r}")
            lines.append(f"IDS    : {ids}")
            lines.append(f"DECODED: {back!r}")
            lines.append(f"LEN    : {len(ids)} tokens")
            lines.append("")
        sample_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[tok] samples written to {sample_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())