"""Model smoke tests — runnable as a script."""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))

import torch
from src.token_to_agent.from_scratch.model import TransformerLM, TransformerConfig, count_params
from src.token_to_agent.from_scratch.model.rope import precompute_rope_cache, apply_rope


def test_model_forward_backward():
    cfg = TransformerConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=64, max_seq_len=16)
    m = TransformerLM(cfg)
    x = torch.randint(0, 128, (2, 8))
    y = torch.randint(0, 128, (2, 8))
    logits, loss = m(x, targets=y)
    assert logits.shape == (2, 8, 128)
    assert loss.requires_grad
    loss.backward()
    # Check that all parameters got gradients.
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
    print(f"[PASS] forward_backward — {count_params(m):,d} params, loss={loss.item():.3f}")


def test_kv_cache_generation():
    cfg = TransformerConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=64, max_seq_len=32)
    m = TransformerLM(cfg)
    m.eval()
    prompt = torch.randint(0, 128, (1, 4))
    out_cache = m.generate(prompt, max_new_tokens=10, use_cache=True)
    out_no = m.generate(prompt, max_new_tokens=10, use_cache=False)
    assert out_cache.shape == (1, 14)
    assert out_no.shape == (1, 14)
    # First 4 tokens should equal the prompt in both cases.
    assert torch.equal(out_cache[0, :4], prompt[0])
    assert torch.equal(out_no[0, :4], prompt[0])
    print("[PASS] kv_cache_generation — both paths return (B, T0+new)")


def test_rope_roundtrip():
    head_dim = 16
    max_seq_len = 32
    cos, sin = precompute_rope_cache(head_dim, max_seq_len, theta=10000.0)
    # Apply RoPE twice should NOT return to identity (rotations don't commute that way),
    # but applying with cos/sin at offset+0 vs offset+T should be different.
    x = torch.randn(1, 4, 1, head_dim)
    y0 = apply_rope(x, cos, sin, offset=0)
    y4 = apply_rope(x, cos, sin, offset=4)
    assert not torch.allclose(y0, y4), "different offsets should produce different outputs"
    print("[PASS] rope_offset_dependence — confirmed position matters")


def test_param_count_breakdown():
    cfg = TransformerConfig(vocab_size=128, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2, d_ff=128, max_seq_len=16)
    m = TransformerLM(cfg)
    breakdown = count_params(m, by_component=True)
    assert "_total" in breakdown
    assert breakdown["token_emb"] > 0
    assert breakdown["blocks"] > 0
    assert breakdown["ln_f"] == 64  # single RMSNorm of d_model=64
    print("[PASS] param_count_breakdown —", breakdown)


if __name__ == "__main__":
    test_model_forward_backward()
    test_kv_cache_generation()
    test_rope_roundtrip()
    test_param_count_breakdown()
    print("\nAll model tests passed.")