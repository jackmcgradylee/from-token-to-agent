"""Custom CUDA/Triton kernels for HW2 Systems.

Components planned:
  - triton_attention.py: Flash-Attention-2-style forward kernel with autotuning
  - reference_attention.py: pure-PyTorch reference (used for correctness tests)
  - interface.py: unified AttentionBackend interface so the model can swap
    "pytorch" / "triton" via config

All kernels target forward pass only for HW2. Backward, fused RMSNorm, fused
MLP, KV cache, torch.compile, FSDP, quantization are EXT-2XX work that
branches off this milestone.
"""

from .interface import (
    AttentionBackend,
    get_backend,
    list_backends,
)

__all__ = ["AttentionBackend", "get_backend", "list_backends"]