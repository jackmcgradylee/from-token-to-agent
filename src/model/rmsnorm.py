"""RMSNorm (Zhang & Sennrich, 2019): y = x / RMS(x) * weight.

RMS(x) = sqrt(mean(x^2) + eps).

This is the LayerNorm replacement used in LLaMA / Mistral / Gemma.
Trainable: per-feature weight (gamma). No bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in fp32 for stability under bf16.
        orig_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_normed = x_fp32 / rms
        return (x_normed.to(orig_dtype) * self.weight)