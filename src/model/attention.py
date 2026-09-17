"""Grouped Query Attention (GQA, Ainslie et al. 2023) with KV cache support.

For HW1 we use PyTorch SDPA as the attention backend. HW2 will swap this
file with a Triton implementation in src/kernels/ -- the *interface* stays
the same: forward(x, cos, sin, mask_or_kv_cache) -> y.

Key feature: KV cache. During inference we don't want to recompute K, V for
already-seen tokens. KVCache stores (K, V) per layer and is appended each
forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class KVCache:
    """Per-layer K, V cache for incremental decoding.

    Shape: K, V are (B, n_kv_heads, T_cache, head_dim). New tokens extend T_cache.
    """

    def __init__(self):
        self.K: torch.Tensor | None = None
        self.V: torch.Tensor | None = None

    def num_cached(self) -> int:
        if self.K is None:
            return 0
        return self.K.shape[2]

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K, V to the cache. Returns the full (K, V) tensors."""
        if self.K is None:
            self.K = k_new
            self.V = v_new
        else:
            self.K = torch.cat([self.K, k_new], dim=2)
            self.V = torch.cat([self.V, v_new], dim=2)
        return self.K, self.V

    def reset(self):
        self.K = None
        self.V = None


class Attention(nn.Module):
    """Multi-head attention with optional grouped-query (n_kv_heads <= n_heads).

    If n_kv_heads == n_heads: standard MHA.
    If n_kv_heads == 1:    multi-query (MQA).
    Else:                  GQA — each KV head is shared by (n_heads / n_kv_heads) Q heads.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads
        if head_dim is None:
            assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
            head_dim = d_model // n_heads
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # Q heads per KV head

        # Single fused Q projection; separate K, V (smaller).
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,         # (B, T, d_model)
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        is_causal: bool = True,
        offset: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Project to (B, T, H, D) then transpose to (B, H, T, D).
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE.
        q = apply_rope(q, cos, sin, offset=offset)
        k = apply_rope(k, cos, sin, offset=offset)

        # Optionally extend KV cache for incremental decoding.
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
            # After cache update, T_total = offset + T (current length).
            T_total = k.shape[2]
        else:
            T_total = offset + T

        # Build attention mask.
        # For training: causal mask on T positions.
        # For incremental decode with kv_cache: positions [0..T_total), and only
        # the last T query positions need to attend to all of them.
        if kv_cache is not None:
            # No mask needed; PyTorch SDPA handles broadcasting.
            attn_mask = None
        else:
            attn_mask = None  # SDPA with is_causal=True is more efficient

        # Repeat KV heads to match Q heads (GQA expansion).
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # PyTorch SDPA. On bf16/fp16 with causal mask, this uses Flash-Attention
        # internally when possible.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal if kv_cache is None else False,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Reshape back: (B, H, T, D) -> (B, T, H, D) -> (B, T, H*D)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)