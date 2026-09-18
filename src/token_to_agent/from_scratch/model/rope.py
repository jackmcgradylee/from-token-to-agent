"""Rotary Positional Embeddings (RoPE, Su et al. 2021).

We precompute the cos/sin cache for [max_seq_len, head_dim] once at module
construction. Forward applies rope in-place to q and k.

`apply_rope` works on the last dim (head_dim), split into pairs (even, odd),
rotate by the precomputed angle for the corresponding position.
"""

from __future__ import annotations

import torch


def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) of shape (max_seq_len, head_dim)."""
    # Compute the inverse-frequency per pair index.
    # For pair i in 0..head_dim/2-1: inv_freq[i] = 1 / theta^(2i / head_dim)
    pair_idx = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq = 1.0 / (theta ** (pair_idx / head_dim))

    # positions 0..max_seq_len-1
    t = torch.arange(max_seq_len, device=device, dtype=dtype)

    # Outer product: (max_seq_len, head_dim/2)
    freqs = torch.outer(t, inv_freq)

    # Duplicate each pair to make it (max_seq_len, head_dim):
    # we use cos/sin in pairs, so the same value is applied to even & odd index.
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(
    x: torch.Tensor,  # (..., head_dim)
    cos: torch.Tensor,  # (max_seq_len, head_dim)
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """Apply rotary embeddings to the last dim of x.

    cos/sin indexed by position (offset, offset+1, ...). We assume x has a
    sequence dimension as its second-to-last dim (so shape (B, T, H, D) or
    (B, H, T, D)); we broadcast along whichever dim corresponds to T.
    """
    # We need cos/sin for positions [offset, offset+T).
    T = x.shape[-2]  # penultimate dim
    if T == 0:
        return x
    cos_slice = cos[offset : offset + T].to(x.device)
    sin_slice = sin[offset : offset + T].to(x.device)

    # Broadcast to x's shape: x has shape (..., T, D), so we add a singleton
    # at the second-to-last position.
    # cos/sin originally (T, D) -> (1, T, D)
    cos_b = cos_slice.unsqueeze(0) if x.dim() == 3 else cos_slice.unsqueeze(0).unsqueeze(0)
    sin_b = sin_slice.unsqueeze(0) if x.dim() == 3 else sin_slice.unsqueeze(0).unsqueeze(0)

    # For a pair (x_even, x_odd), rotation is:
    #   [cos, -sin] [x_even]
    #   [sin,  cos] [x_odd]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    # Rotate.
    out_even = x_even * cos_b[..., 0::2] - x_odd * sin_b[..., 0::2]
    out_odd = x_odd * cos_b[..., 0::2] + x_even * sin_b[..., 0::2]

    # Interleave back.
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out