"""Model utilities (param count, etc.)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import TransformerConfig, TransformerLM


def count_params(model: nn.Module, by_component: bool = False) -> int | dict:
    """Total trainable params, or breakdown by top-level component.

    Useful for the HW1 DoD table.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not by_component:
        return total

    breakdown: dict[str, int] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Top-level component is the first segment of the name.
        comp = name.split(".")[0]
        breakdown[comp] = breakdown.get(comp, 0) + p.numel()
    breakdown["_total"] = total
    return breakdown


def model_from_config(cfg_dict: dict) -> TransformerLM:
    """Instantiate a model from a plain dict (e.g., loaded from YAML)."""
    cfg = TransformerConfig(**cfg_dict)
    return TransformerLM(cfg)