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

W4: also exposes an autograd.Function wrapper so the reference can be used
as a differentiable op (for correctness testing against the Triton backward
kernel). The functional form returns (out, save_ctx) for backward, matching
what `torch.autograd.Function.apply` expects.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def reference_attention_forward(
    q: torch.Tensor,           # (B, H, T, D)  already RoPE-applied
    k: torch.Tensor,           # (B, H_kv, S, D) already RoPE-applied
    v: torch.Tensor,           # (B, H_kv, S, D)
    is_causal: bool = True,
    softmax_scale: Optional[float] = None,
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


# ---------------------------------------------------------------------------
# Naive PyTorch attention (no SDPA fusion).
#
# Pure matmul, bias, softmax, matmul. Used as the "naive" baseline in W4
# benchmarks. We keep it explicitly written out (not a one-liner) so the
# FLOPs can be counted clearly.
# ---------------------------------------------------------------------------


def naive_attention_forward(
    q: torch.Tensor,           # (B, H, T, D)
    k: torch.Tensor,           # (B, H, S, D)
    v: torch.Tensor,           # (B, H, S, D)
    is_causal: bool = True,
    softmax_scale: Optional[float] = None,
    n_rep: int = 1,
) -> torch.Tensor:
    """Pure-PyTorch attention, no SDPA. Always differentiable.

    Identical math to `reference_attention_forward` but never reaches into
    F.scaled_dot_product_attention — explicit matmul + softmax + matmul.
    Useful as a CPU-runnable baseline for benchmarking memory and latency
    against SDPA / Triton. Returns a tensor that participates in autograd
    via standard PyTorch ops.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    T = q.shape[2]
    S = k.shape[2]

    scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, S)
    scores = scores * softmax_scale
    if is_causal and T == S:
        mask = torch.triu(
            torch.full((T, S), float("-inf"), device=q.device, dtype=q.dtype),
            diagonal=1,
        )
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


# ---------------------------------------------------------------------------
# Autograd Function wrapper — used by `autograd.gradcheck` and by the
# Triton-side autograd.Function to verify gradient equivalence.
# ---------------------------------------------------------------------------


class ReferenceAttention(torch.autograd.Function):
    """Custom autograd.Function for reference attention.

    Forward stores (Q, K, V, attn_weights, softmax_scale, is_causal, n_rep).
    Backward computes dQ, dK, dV via the standard 4-line derivation:

        D = (dO * O).sum(-1, keepdim=True)              # (B, H, T, 1)
        dA = (dO @ V^T)                                 # (B, H, T, S)
        dP = dA * attn_weights                          # (B, H, T, S)
        dS = dP - D * attn_weights                      # (B, H, T, S)
        dQ = (dS * scale) @ K                           # (B, H, T, D)
        dK = (dS^T * scale) @ Q                         # (B, H_kv, S, D)
        dV = attn_weights^T @ dO                        # (B, H_kv, S, D)

    If GQA is active (n_rep > 1), the dQ stays per-H, but dK/dV have to
    be reduced across the Q heads that share each KV head.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        softmax_scale: float,
        n_rep: int,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
        if n_rep > 1:
            k_eff = k.repeat_interleave(n_rep, dim=1)
            v_eff = v.repeat_interleave(n_rep, dim=1)
        else:
            k_eff = k
            v_eff = v
        T = q.shape[2]
        S = k_eff.shape[2]
        scores = torch.matmul(q, k_eff.transpose(-2, -1)) * softmax_scale
        if is_causal and T == S:
            mask = torch.triu(
                torch.full((T, S), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=1,
            )
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v_eff)
        ctx.save_for_backward(q, k, v, weights)
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.n_rep = n_rep
        return out

    @staticmethod
    def backward(ctx, d_out: torch.Tensor):
        q, k, v, weights = ctx.saved_tensors
        is_causal = ctx.is_causal
        softmax_scale = ctx.softmax_scale
        n_rep = ctx.n_rep

        # Expand K/V if GQA so backward can use the same shapes as forward.
        if n_rep > 1:
            k_eff = k.repeat_interleave(n_rep, dim=1)
            v_eff = v.repeat_interleave(n_rep, dim=1)
        else:
            k_eff = k
            v_eff = v

        # If a causal mask was applied in forward, zeros-out gradient flow
        # to masked positions. Cheap: just zero the appropriate entries of
        # dP (derived below).
        T = q.shape[2]
        S = k_eff.shape[2]

        # D = rowsum(d_out * out)
        D = (d_out * torch.matmul(weights, v_eff)).sum(dim=-1, keepdim=True)
        # dP = d_out @ V^T  (pointwise * attn)
        dA = torch.matmul(d_out, v_eff.transpose(-2, -1))
        dP = dA * weights
        # dS = dP - weights * D
        dS = dP - weights * D
        dS = dS * softmax_scale

        if is_causal and T == S:
            # Mask out positions that were -inf in forward.
            mask = torch.triu(
                torch.ones(T, S, device=q.device, dtype=q.dtype),
                diagonal=1,
            ).bool()
            dS = dS.masked_fill(mask, 0.0)

        # dQ = dS @ K
        dQ = torch.matmul(dS, k_eff)
        # dK_eff = dS^T @ Q
        dK_eff = torch.matmul(dS.transpose(-2, -1), q)
        # dV_eff = weights^T @ d_out
        dV_eff = torch.matmul(weights.transpose(-2, -1), d_out)

        # Reduce across the Q-head dim that shared each KV head (GQA only).
        if n_rep > 1:
            H = q.shape[1]
            H_kv = k.shape[1]
            # dK_eff, dV_eff are (B, H, S, D); collapse H -> H_kv by reshape-sum.
            dK = dK_eff.view(q.shape[0], H_kv, n_rep, S, q.shape[-1]).sum(dim=2)
            dV = dV_eff.view(q.shape[0], H_kv, n_rep, S, q.shape[-1]).sum(dim=2)
        else:
            dK = dK_eff
            dV = dV_eff

        return dQ, dK, dV, None, None, None


def reference_attention_forward_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    softmax_scale: Optional[float] = None,
    n_rep: int = 1,
) -> torch.Tensor:
    """Autograd-aware reference attention. Used by autograd.gradcheck."""
    return ReferenceAttention.apply(q, k, v, is_causal, softmax_scale, n_rep)