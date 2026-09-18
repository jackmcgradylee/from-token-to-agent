"""Transformer LM, written from scratch.

Components implemented:
  - RMSNorm (default; LLaMA / Mistral style)
  - LayerNorm (W2 ablation; Ba et al. 2016)
  - Rotary Positional Embeddings (RoPE)
  - Grouped Query Attention (GQA) with KV cache support
    (set n_kv_heads < n_heads to use GQA; n_kv_heads == n_heads is MHA)
  - SwiGLU MLP
  - Truncated-normal initialization (GPT-2 style)
  - Causal LM head tied with input embeddings

W2 ablations (run via experiments/w02/exp-001-rmsnorm-vs-layernorm
and experiments/w02/exp-002-mha-vs-gqa):
  - LayerNorm vs RMSNorm: same architecture, swap norm kind via
    cfg.norm_kind, count params, compare forward latency / output.
  - MHA vs GQA: same architecture, set n_kv_heads to 4 (GQA-4) vs
    12 (MHA), measure KV cache size and forward latency.
"""

from .model import TransformerLM, TransformerConfig
from .rmsnorm import RMSNorm
from .layernorm import LayerNorm
from .rope import precompute_rope_cache, apply_rope
from .swiglu import SwiGLU
from .attention import Attention, KVCache
from .utils import count_params

__all__ = [
    "TransformerLM",
    "TransformerConfig",
    "RMSNorm",
    "LayerNorm",
    "precompute_rope_cache",
    "apply_rope",
    "SwiGLU",
    "Attention",
    "KVCache",
    "count_params",
]