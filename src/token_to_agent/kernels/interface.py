"""Unified attention backend interface.

The model picks the backend via `attn_backend` in its config:
  - "pytorch":  uses F.scaled_dot_product_attention (HW1 default).
  - "triton":   uses src/kernels/triton_attention.triton_attention_forward.
  - "reference": uses src/kernels/reference_attention.reference_attention_forward
                 (slowest, used as oracle for correctness tests).

On machines without CUDA, requests for "triton" automatically fall back to
"reference" with a logged warning. This keeps the model.runnable everywhere
while still showing what the speedup would be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from . import reference_attention
from . import triton_attention as _triton_mod


Backend = Literal["pytorch", "triton", "reference"]


@dataclass
class AttentionBackend:
    name: str
    available: bool
    reason: str  # human-readable "why available / why not"


def _probe_pytorch() -> AttentionBackend:
    # F.scaled_dot_product_attention works on CPU and CUDA.
    return AttentionBackend(name="pytorch", available=True, reason="F.scaled_dot_product_attention always works")


def _probe_triton() -> AttentionBackend:
    if not _triton_mod.is_available():
        if not torch.cuda.is_available():
            return AttentionBackend(
                name="triton",
                available=False,
                reason="CUDA not available (Jetson / CPU-only host). Falling back to reference.",
            )
        return AttentionBackend(
            name="triton",
            available=False,
            reason="triton import failed (likely pip issue).",
        )
    return AttentionBackend(name="triton", available=True, reason="Triton + CUDA ready")


def list_backends() -> list[AttentionBackend]:
    return [_probe_pytorch(), _probe_triton()]


def get_backend(requested: str) -> AttentionBackend:
    """Resolve a backend name to a backend descriptor."""
    requested = requested.lower()
    if requested == "pytorch":
        return _probe_pytorch()
    if requested == "triton":
        return _probe_triton()
    if requested == "reference":
        return AttentionBackend(name="reference", available=True, reason="pure-PyTorch reference oracle")
    raise ValueError(f"unknown backend {requested!r}; expected one of pytorch/triton/reference")


def call(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    n_rep: int = 1,
    training: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Dispatch attention forward to the requested backend.

    For "triton" on a CUDA-less host, automatically falls back to reference.
    """
    if backend == "pytorch":
        import torch.nn.functional as F
        # GQA expansion: PyTorch SDPA doesn't support GQA natively in 2.5, so
        # repeat KV heads if n_rep > 1.
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        # SDPA expects (B, H, T, D); shape already matches.
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=is_causal,
            dropout_p=dropout_p if training else 0.0,
        )

    if backend == "triton":
        if _triton_mod.is_available():
            return _triton_mod.triton_attention_forward(
                q, k, v,
                is_causal=is_causal,
                softmax_scale=softmax_scale,
                n_rep=n_rep,
            )
        # Silent fallback (caller logs).
        return reference_attention.reference_attention_forward(
            q, k, v,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            n_rep=n_rep,
        )

    if backend == "reference":
        return reference_attention.reference_attention_forward(
            q, k, v,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            n_rep=n_rep,
        )

    raise ValueError(f"unknown backend {backend!r}")