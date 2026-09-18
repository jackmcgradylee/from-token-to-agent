"""Tests for kv_cache.py math (pure-Python)."""

from __future__ import annotations

import pytest

from src.token_to_agent.systems.inference.kv_cache import (
    PRESETS,
    cost_per_million_tokens,
    dtype_bytes,
    kv_bytes_for_request,
    kv_bytes_per_token,
    kv_bytes_per_token_per_layer,
    max_concurrent_at_seq_len,
    model_weight_bytes,
)


def test_dtype_bytes_table():
    assert dtype_bytes("fp32") == 4
    assert dtype_bytes("fp16") == 2
    assert dtype_bytes("bf16") == 2
    assert dtype_bytes("int8") == 1
    with pytest.raises(ValueError):
        dtype_bytes("fp64")


def test_llama2_7b_per_token():
    """Llama-2-7B is MHA: 2*32*128*2 = 16 KB per layer per token."""
    cfg = PRESETS["llama2-7b"]
    per_layer = kv_bytes_per_token_per_layer(cfg, "fp16")
    assert per_layer == 2 * 32 * 128 * 2  # 16384 bytes = 16 KB
    assert per_layer == 16384
    # 32 layers => 512 KB / token.
    per_token = kv_bytes_per_token(cfg, "fp16")
    assert per_token == 32 * 16384
    assert per_token == 524288  # 512 KB exactly


def test_mistral_7b_per_token_gqa_savings():
    """Mistral-7B uses GQA-8: 4 KB per layer per token, 128 KB total."""
    cfg = PRESETS["mistral-7b"]
    per_token = kv_bytes_per_token(cfg, "fp16")
    # 2 * 8 * 128 * 2 = 4096 bytes/layer, * 32 layers = 131072 = 128 KB
    assert per_token == 131072


def test_per_request_scales_linearly():
    cfg = PRESETS["llama2-7b"]
    assert kv_bytes_for_request(cfg, 0, "fp16") == 0
    assert kv_bytes_for_request(cfg, 1, "fp16") == kv_bytes_per_token(cfg, "fp16")
    assert kv_bytes_for_request(cfg, 4096, "fp16") == 4096 * kv_bytes_per_token(cfg, "fp16")


def test_seq_len_bounds():
    cfg = PRESETS["llama2-7b"]
    with pytest.raises(ValueError):
        kv_bytes_for_request(cfg, -1, "fp16")
    with pytest.raises(ValueError):
        kv_bytes_for_request(cfg, cfg.max_seq_len + 1, "fp16")


def test_dtype_affects_size():
    cfg = PRESETS["llama2-7b"]
    fp32 = kv_bytes_for_request(cfg, 1024, "fp32")
    fp16 = kv_bytes_for_request(cfg, 1024, "fp16")
    int8 = kv_bytes_for_request(cfg, 1024, "int8")
    assert fp32 == 2 * fp16
    assert fp16 == 2 * int8


def test_model_weight_bytes_scales_with_params():
    cfg = PRESETS["llama2-7b"]
    weights_fp16 = model_weight_bytes(cfg, "fp16")
    assert weights_fp16 == int(6.7 * 1e9) * 2
    assert abs(weights_fp16 - 13.4e9) < 1e7  # ~13.4 GB


def test_max_concurrent_at_seq_len():
    cfg = PRESETS["llama2-7b"]
    h100 = 80 * 1024**3
    n = max_concurrent_at_seq_len(cfg, 4096, h100, "fp16", 2.0)
    # 60 GB budget after weights (~13.4 GB) + overhead (2 GB).
    # Each req at 4096 = 2 GB. Should fit ~32.
    assert 28 <= n <= 36


def test_max_concurrent_zero_when_overhead_consumes_all():
    cfg = PRESETS["llama2-7b"]
    tiny_gpu = 1024  # 1 KB
    n = max_concurrent_at_seq_len(cfg, 4096, tiny_gpu, "fp16", 2.0)
    assert n == 0


def test_cost_per_million_tokens_sanity():
    # $2/hr at 1000 TPS: $/token = 2/3600/1000; * 1e6 = 0.556
    assert abs(cost_per_million_tokens(2.0, 1000.0) - (2.0 * 1e6 / (3600 * 1000))) < 1e-6
    # $0 if free hardware (but tps > 0)
    assert cost_per_million_tokens(0.0, 1000.0) == 0.0
    with pytest.raises(ValueError):
        cost_per_million_tokens(2.0, 0.0)


def test_all_presets_have_consistent_shapes():
    """n_heads must be a multiple of n_kv_heads; both > 0."""
    for name, cfg in PRESETS.items():
        assert cfg.n_heads > 0
        assert cfg.n_kv_heads > 0
        assert cfg.n_heads % cfg.n_kv_heads == 0, f"{name}: GQA ratio not integer"
        assert cfg.head_dim > 0
        assert cfg.max_seq_len > 0
