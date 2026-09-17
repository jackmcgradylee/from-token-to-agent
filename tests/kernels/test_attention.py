"""Triton attention correctness test.

Compares `triton_attention_forward` (when CUDA is available) against the
pure-PyTorch reference implementation. Skipped on CUDA-less hosts with a
logged skip rather than failing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))

import torch

from src.token_to_agent.kernels import reference_attention
from src.token_to_agent.kernels import triton_attention as triton_mod
from src.token_to_agent.kernels import interface as attn_iface


def test_triton_vs_reference():
    if not triton_mod.is_available():
        print("[SKIP] triton_attention not available (no CUDA)")
        return
    print("[triton available, running correctness checks]")

    cases = [
        # (B, H, H_kv, T, D)
        (1, 4, 1, 64, 32),     # tiny
        (1, 4, 4, 128, 32),    # no GQA, larger
        (1, 8, 2, 64, 64),     # GQA, head_dim=64
        (2, 4, 1, 128, 32),    # batched, MQA
    ]
    dtype = torch.float32  # use fp32 to avoid bf16 noise in correctness
    atol = 1e-3
    rtol = 1e-3

    for B, H, H_kv, T, D in cases:
        torch.manual_seed(B * 100 + H * 10 + T)
        q = torch.randn(B, H, T, D, device="cuda", dtype=dtype) * 0.5
        k = torch.randn(B, H_kv, T, D, device="cuda", dtype=dtype) * 0.5
        v = torch.randn(B, H_kv, T, D, device="cuda", dtype=dtype) * 0.5
        n_rep = H // H_kv

        ref = reference_attention.reference_attention_forward(
            q, k, v, is_causal=True, n_rep=n_rep,
        )
        tri = triton_mod.triton_attention_forward(
            q, k, v, is_causal=True, n_rep=n_rep,
        )
        max_err = (tri - ref).abs().max().item()
        rel_err = max_err / max(ref.abs().max().item(), 1e-8)
        assert torch.allclose(tri, ref, atol=atol, rtol=rtol), (
            f"Triton vs reference diverged: B={B} H={H} H_kv={H_kv} T={T} D={D} "
            f"max_abs_err={max_err:.3e} rel_err={rel_err:.3e}"
        )
        print(f"  [PASS] B={B} H={H} H_kv={H_kv} T={T} D={D} max_err={max_err:.2e}")


def test_backend_dispatch():
    """Interface should silently fall back when triton unavailable."""
    backends = attn_iface.list_backends()
    print("[PASS] backend dispatch:")
    for b in backends:
        print(f"        {b.name:10s} available={b.available} reason={b.reason}")

    # reference should always work.
    B, H, H_kv, T, D = 1, 4, 1, 32, 16
    q = torch.randn(B, H, T, D, dtype=torch.float32) * 0.5
    k = torch.randn(B, H_kv, T, D, dtype=torch.float32) * 0.5
    v = torch.randn(B, H_kv, T, D, dtype=torch.float32) * 0.5
    out = attn_iface.call("reference", q, k, v, is_causal=True, n_rep=H // H_kv)
    assert out.shape == (B, H, T, D)
    print("[PASS] reference backend runs on CPU")


def test_attention_layer_backend_switch():
    """Verify the model's Attention.set_backend routes correctly."""
    from src.token_to_agent.model import TransformerLM, TransformerConfig
    cfg = TransformerConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=64, max_seq_len=32)
    m = TransformerLM(cfg)
    for b in ["reference", "pytorch", "triton"]:
        m.set_attn_backend(b)
        # Verify the layer actually switched.
        layer_backend = m.blocks[0].attn.attn_backend
        info = attn_iface.get_backend(b)
        expected = b if info.available else ("reference" if b == "triton" else "pytorch")
        assert layer_backend == expected, (
            f"set_attn_backend({b!r}) -> layer backend {layer_backend!r}, "
            f"expected {expected!r}"
        )
    print("[PASS] Attention.set_backend routes through fallback correctly")


if __name__ == "__main__":
    test_backend_dispatch()
    test_attention_layer_backend_switch()
    test_triton_vs_reference()
    print("\nAll Triton/attention tests passed.")