"""Transformer LM, written from scratch.

Components implemented:
  - RMSNorm
  - Rotary Positional Embeddings (RoPE)
  - Grouped Query Attention (GQA) with KV cache support
  - SwiGLU MLP
  - Truncated-normal initialization (GPT-2 style)
  - Causal LM head tied with input embeddings

HW2 will swap out src/model/attention.py with a Triton kernel, but keep
every other component. That's the whole point of keeping `src/` as a growing
codebase instead of copying.
"""

from .model import TransformerLM, TransformerConfig
from .rmsnorm import RMSNorm
from .rope import precompute_rope_cache, apply_rope
from .swiglu import SwiGLU
from .attention import Attention, KVCache
from .utils import count_params

__all__ = [
    "TransformerLM",
    "TransformerConfig",
    "RMSNorm",
    "precompute_rope_cache",
    "apply_rope",
    "SwiGLU",
    "Attention",
    "KVCache",
    "count_params",
]