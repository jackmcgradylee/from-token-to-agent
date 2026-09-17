"""Cosine LR schedule with linear warmup, the LLaMA standard."""

from __future__ import annotations

import math


def cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup -> cosine decay to min_lr_ratio * base_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cos)