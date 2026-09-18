"""systems.inference — Inference-time math.

- kv_cache: KV cache memory model, max-concurrent estimator, $/1M-token math.
"""
from .kv_cache import (
    PRESETS,
    ModelConfig,
    DType,
    dtype_bytes,
    kv_bytes_per_token,
    kv_bytes_per_token_per_layer,
    kv_bytes_for_request,
    model_weight_bytes,
    max_concurrent_at_seq_len,
    cost_per_million_tokens,
)

__all__ = [
    "PRESETS",
    "ModelConfig",
    "DType",
    "dtype_bytes",
    "kv_bytes_per_token",
    "kv_bytes_per_token_per_layer",
    "kv_bytes_for_request",
    "model_weight_bytes",
    "max_concurrent_at_seq_len",
    "cost_per_million_tokens",
]
