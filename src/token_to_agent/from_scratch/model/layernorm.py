"""LayerNorm (Ba et al. 2016): y = (x - mean) / sqrt(var + eps) * weight + bias.

Standard pre-norm placement for Transformer blocks.

This module exists for W2 ablation: a controlled comparison of
LayerNorm vs RMSNorm in the same architecture. Most modern open-source
LLMs (LLaMA, Mistral, Gemma) use RMSNorm because it drops the mean
computation and the bias term — fewer ops, fewer parameters, similar
training stability for autoregressive language modeling.

References:
  - Ba, Kiros & Hinton 2016, "Layer Normalization".
  - Zhang & Sennrich 2019, "Root Mean Square Layer Normalization"
    (RMSNorm — the modern alternative used here by default).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Standard LayerNorm with learned per-feature affine (gamma, beta).

    Computed in fp32 for stability under bf16, identical pattern to
    RMSNorm in this repo.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = x_fp32.var(dim=-1, keepdim=True, unbiased=False)
        x_normed = (x_fp32 - mean) / torch.sqrt(var + self.eps)
        out = x_normed * self.weight + self.bias
        return out.to(orig_dtype)