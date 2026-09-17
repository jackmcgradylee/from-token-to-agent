"""Tiny self-contained text corpus for tokenizer + toy LM training.

This file is intentionally a placeholder for HW1's small-text dataset. The
real pipeline (HW3) will live in src/data/ and load TinyStories /
Wikipedia / etc.

For now we generate a simple synthetic corpus so the pipeline can be
exercised end-to-end on Jetson without downloads.

Usage:
    python scripts/make_sample_corpus.py \\
        --out data/raw/sample.txt \\
        --size-mb 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


SUBJECTS = [
    "Lily", "Tom", "the little rabbit", "the cat", "the brave knight",
    "the dragon", "a small bird", "the wizard", "the boy", "the girl",
    "the fox", "the turtle", "a tiny mouse", "the prince", "the princess",
]

PLACES = [
    "in the forest", "by the sea", "in a small village", "in the garden",
    "on the mountain", "in the castle", "in the meadow", "by the river",
    "under the old tree", "in the dark cave",
]

ACTIONS = [
    "found a shiny stone",
    "saw a strange light",
    "met a talking owl",
    "heard a soft voice",
    "picked a red flower",
    "made a new friend",
    "lost her way home",
    "discovered a hidden door",
    "learned a magic word",
    "saw something wonderful",
]

CONNECTORS = [
    "Once upon a time, ", "And then, ", "Soon after, ", "Later that day, ",
    "When the sun set, ", "The next morning, ", "Suddenly, ", "Quietly, ",
    "Without warning, ", "Happily, ",
]

ENDINGS = [
    "and they lived happily ever after.",
    "the end.",
    "and went home for dinner.",
    "and never forgot that day.",
    "and became best friends forever.",
    "and learned something important.",
    "and fell asleep smiling.",
    "and that was just the beginning.",
]


def make_sentence(rng: random.Random) -> str:
    s = (
        rng.choice(CONNECTORS)
        + rng.choice(SUBJECTS)
        + " "
        + rng.choice(ACTIONS)
        + " "
        + rng.choice(PLACES)
        + ". "
        + rng.choice(ENDINGS)
    )
    return s


def make_paragraph(rng: random.Random, n_sentences: int = 5) -> str:
    return " ".join(make_sentence(rng) for _ in range(n_sentences))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--size-mb", type=int, default=10,
                   help="Approx target size in MB.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_bytes = args.size_mb * 1024 * 1024
    written = 0
    line_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        while written < target_bytes:
            para = make_paragraph(rng, n_sentences=rng.randint(3, 8))
            line = para + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))
            line_count += 1

    print(f"[corpus] wrote {line_count} lines / ~{written / (1024*1024):.1f} MB to {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())