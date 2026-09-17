"""Pure-PyTorch reference attention, used as the correctness oracle.

This file is the ground truth. Triton's job is to match its outputs within
tolerance (max abs error < 5e-3 for fp16/bf16, < 1e-5 for fp32) while being
substantially faster.

Why a separate file:
  - Unit tests compare `triton_attention.forward(...)` to this reference on
    small inputs that both can run.
  - Benchmark numbers reported in HW2 use this implementation as the
    "PyTorch baseline" row.
  - On machines without CUDA (e.g., this Jetson), this is the only path
    that runs.

Supports:
  - causal masking
  - grouped-query attention (n_kv_heads < n_heads)
  - KV cache append (incremental decoding)
  - optional RoPE applied externally (kernel just does QK^T + softmax + @V)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reference_attention_forward(
    q: torch.Tensor,           # (B, H, T, D)  already RoPE-applied
    k: torch.Tensor,           # (B, H_kv, S, D) already RoPE-applied
    v: torch.Tensor,           # (B, H_kv, S, D)
    is_causal: bool = True,
    softmax_scale: float | None = None,
    dropout_p: float = 0.0,
    n_rep: int = 1,            # Q heads per KV head (1 = MHA)
) -> torch.Tensor:
    """Standard scaled-dot-product attention with optional GQA expansion.

    Returns (B, H, T, D).
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Repeat KV heads if GQA.
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    # (B, H, T, D) @ (B, H, D, S) -> (B, H, T, S)
    T = q.shape[2]
    S = k.shape[2]
    attn = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    if is_causal:
        # Only mask when query and key have the same length (training-style).
        # For incremental decode (T << S), is_causal is False (full visibility).
        if T == S:
            mask = torch.triu(
                torch.full((T, S), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=1,
            )
            attn = attn + mask

    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=True)

    out = torch.matmul(attn, v)  # (B, H, T, D)
    return out