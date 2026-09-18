"""Generate a deterministic ~1 MB synthetic TinyStories-style corpus.

Used by all W3 scaling-pilot experiments to satisfy the iso-data
condition. Each model in the sweep trains on the *same* tokenized
sequence, so the only variable that changes is model capacity.

Run:
    PYTHONPATH=. .venv/bin/python scripts/make_sample_corpus.py \
        --out data/raw/sample.txt --target-bytes 1048576
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Toy character + object + setting templates. Deterministic shuffle so
# that the same seed always reproduces the same corpus. Output looks
# vaguely like TinyStories: short sentences, simple vocabulary, repetitive
# structure.

CHARACTERS = [
    "Lily", "Ben", "Mia", "Sam", "Ella", "Jack", "Rose", "Tom", "Ava", "Noah",
    "the little boy", "the little girl", "the big dog", "the small cat",
    "a curious child", "a brave knight", "a tiny mouse", "a friendly dragon",
]
PLACES = [
    "in the garden", "at the pond", "in the forest", "at the castle",
    "by the river", "in a small house", "on a hill", "near the old oak tree",
    "under the big bridge", "in the kitchen",
]
ACTIONS = [
    "found a shiny pebble",
    "saw a colorful butterfly",
    "planted a tiny seed",
    "made a paper boat",
    "heard a strange sound",
    "caught a falling leaf",
    "shared a piece of bread",
    "fixed a broken toy",
    "discovered a hidden door",
    "wrote a small note",
]
ENDINGS = [
    "and smiled.", "and felt happy.", "and laughed out loud.",
    "and ran home to tell everyone.",
    "and learned something new.",
    "and promised to come back tomorrow.",
    "and waved goodbye.",
    "and was very proud.",
]


def make_paragraph(rng: random.Random) -> str:
    chars = rng.choice(CHARACTERS)
    place = rng.choice(PLACES)
    action = rng.choice(ACTIONS)
    ending = rng.choice(ENDINGS)
    return f"{chars} {action} {place}, {ending}"


def make_corpus(target_bytes: int, seed: int = 42) -> str:
    rng = random.Random(seed)
    parts: list[str] = []
    total = 0
    while total < target_bytes:
        para = make_paragraph(rng) + "\n\n"
        parts.append(para)
        total += len(para)
    text = "".join(parts)
    return text[:target_bytes]


def main() -> int:
    p = argparse.ArgumentParser(description="Generate deterministic TinyStories-style corpus")
    p.add_argument("--out", default="data/raw/sample.txt",
                   help="Output path (relative to repo root).")
    p.add_argument("--target-bytes", type=int, default=1_048_576,
                   help="Target size in bytes (default 1 MiB).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = make_corpus(args.target_bytes, seed=args.seed)
    out.write_text(text, encoding="utf-8")
    print(f"[make_corpus] wrote {len(text):,} bytes to {out}  (seed={args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())