"""SwiGLU MLP (Shazeer 2020): the gating FFN used in LLaMA / PaLM.

SwiGLU(x) = (silu(W1 x) * W2 x) @ W3

We use a single fused weight W_gate (containing W1 and W2 stacked) to halve
the kernel launches. Internally it splits into gate and up, applies silu and
multiplies, then projects down.

Parameter dims:
  in_dim  = d_model
  hidden = approx (2/3) * 4 * d_model, rounded to a multiple of 64 for kernel
           efficiency. (LLaMA-style.)
  out_dim = d_model
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _swiglu_hidden(d_model: int, multiple_of: int = 64) -> int:
    """LLaMA-style hidden dim: 2/3 * 4 * d_model, rounded up to multiple_of."""
    hidden = int(2 * d_model * 4 / 3)
    hidden = ((hidden + multiple_of - 1) // multiple_of) * multiple_of
    return hidden


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = _swiglu_hidden(d_model)
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        # Fused gate + up projection for efficiency.
        self.gate_up_proj = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)