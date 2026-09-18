"""W2 tests for architecture ablations: LayerNorm vs RMSNorm, MHA vs GQA.

Run from repo root:
    PYTHONPATH=. .venv/bin/python tests/model/test_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from src.token_to_agent.from_scratch.model import (  # noqa: E402
    Attention,
    KVCache,
    LayerNorm,
    RMSNorm,
    TransformerConfig,
    TransformerLM,
    count_params,
)


# ----- tiny test harness -----------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, msg: str = "") -> None:
    _results.append((name, ok, msg))
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {name}{(' — ' + msg) if msg else ''}")


def _make_cfg(**overrides) -> TransformerConfig:
    cfg = TransformerConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=None,
        head_dim=None,
        d_ff=None,
        max_seq_len=32,
        norm_eps=1e-5,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ----- LayerNorm vs RMSNorm mathematical sanity -----------------------

def test_layernorm_zero_weight_equals_zero_centered_norm() -> None:
    """LayerNorm with weight=1, bias=0: output has mean ~0 and std ~1."""
    torch.manual_seed(0)
    x = torch.randn(8, 16) * 5 + 3
    ln = LayerNorm(16).eval()
    with torch.no_grad():
        y = ln(x)
    _check(
        "layernorm.output_mean~0",
        y.mean(dim=-1).abs().max().item() < 1e-4,
        f"max |mean|={y.mean(dim=-1).abs().max().item():.2e}",
    )
    _check(
        "layernorm.output_std~1",
        (y.std(dim=-1, unbiased=False) - 1.0).abs().max().item() < 1e-4,
        f"max |std-1|={(y.std(dim=-1, unbiased=False) - 1.0).abs().max().item():.2e}",
    )


def test_rmsnorm_zero_weight_equals_unit_rms() -> None:
    """RMSNorm with weight=1, bias=N/A: output has RMS ~1."""
    torch.manual_seed(0)
    x = torch.randn(8, 16) * 5 + 3
    rn = RMSNorm(16).eval()
    with torch.no_grad():
        y = rn(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    _check(
        "rmsnorm.output_rms~1",
        (rms - 1.0).abs().max().item() < 1e-4,
        f"max |rms-1|={(rms - 1.0).abs().max().item():.2e}",
    )


def test_layernorm_has_more_parameters_than_rmsnorm() -> None:
    """LayerNorm has 2*dim params (weight+bias); RMSNorm has 1*dim (weight only)."""
    ln = LayerNorm(64)
    rn = RMSNorm(64)
    _check(
        "layernorm.params == 2*dim",
        sum(p.numel() for p in ln.parameters()) == 128,
    )
    _check(
        "rmsnorm.params == dim",
        sum(p.numel() for p in rn.parameters()) == 64,
    )


# ----- TransformerConfig.norm_kind wiring ------------------------------

def test_norm_kind_rmsnorm_default() -> None:
    """Default norm_kind is 'rmsnorm' for backward compatibility."""
    cfg = _make_cfg()
    _check("default.norm_kind == rmsnorm", cfg.norm_kind == "rmsnorm", f"got {cfg.norm_kind!r}")


def test_norm_kind_layernorm_builds_with_bias() -> None:
    """norm_kind='layernorm' produces blocks with LayerNorm modules (with bias)."""
    cfg = _make_cfg(norm_kind="layernorm")
    model = TransformerLM(cfg)
    block0 = model.blocks[0]
    has_bias = isinstance(block0.ln1, LayerNorm) and block0.ln1.bias is not None
    _check(
        "norm_kind=layernorm -> blocks.ln1 is LayerNorm with bias",
        has_bias,
    )


def test_norm_kind_rmsnorm_builds_without_bias() -> None:
    """norm_kind='rmsnorm' produces blocks with RMSNorm modules (no bias)."""
    cfg = _make_cfg(norm_kind="rmsnorm")
    model = TransformerLM(cfg)
    block0 = model.blocks[0]
    no_bias = isinstance(block0.ln1, RMSNorm)
    _check(
        "norm_kind=rmsnorm -> blocks.ln1 is RMSNorm",
        no_bias,
    )


def test_layernorm_model_has_more_params_than_rmsnorm_model() -> None:
    """2-norm model and rmsnorm model differ only in norm — LayerNorm has more params."""
    cfg = _make_cfg()
    torch.manual_seed(0)
    m_rms = TransformerLM(_make_cfg(norm_kind="rmsnorm"))
    torch.manual_seed(0)
    m_ln = TransformerLM(_make_cfg(norm_kind="layernorm"))
    # 2 norms per block * n_layers * d_model extra bias params,
    # PLUS ln_f (final norm, 1 extra bias of dim).
    # Total: 2 * n_layers * d_model + 1 * d_model.
    # With n_layers=2, d_model=64: 256 + 64 = 320.
    expected_extra = (2 * cfg.n_layers + 1) * cfg.d_model
    diff = count_params(m_ln) - count_params(m_rms)
    _check(
        "layernorm_model has exactly the expected param delta over rmsnorm_model",
        diff == expected_extra,
        f"diff={diff} expected={expected_extra}",
    )


def test_norm_ablation_forward_works() -> None:
    """Both norm kinds produce well-shaped tensors with finite values."""
    torch.manual_seed(0)
    for kind in ("rmsnorm", "layernorm"):
        cfg = _make_cfg(norm_kind=kind)
        m = TransformerLM(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            logits, _ = m(ids)
        _check(
            f"norm={kind}: forward returns finite logits",
            torch.isfinite(logits).all().item(),
        )
        _check(
            f"norm={kind}: logits shape == (B, T, V)",
            tuple(logits.shape) == (2, 16, cfg.vocab_size),
        )


# ----- MHA vs GQA ---------------------------------------------------

def test_mha_n_kv_heads_equals_n_heads() -> None:
    """n_kv_heads == n_heads -> MHA, KV projection has full head count."""
    torch.manual_seed(0)
    cfg = _make_cfg(n_heads=4, n_kv_heads=4)
    m = TransformerLM(cfg)
    block = m.blocks[0]
    n_kv_total = cfg.n_kv_heads * (cfg.d_model // cfg.n_heads)
    _check(
        "MHA: k_proj out_features uses n_kv_heads == n_heads",
        block.attn.k_proj.out_features == n_kv_total,
        f"got {block.attn.k_proj.out_features}",
    )


def test_gqa_n_kv_heads_less_than_n_heads() -> None:
    """n_kv_heads < n_heads -> GQA; KV projection is smaller."""
    torch.manual_seed(0)
    cfg = _make_cfg(n_heads=4, n_kv_heads=2)
    m = TransformerLM(cfg)
    block = m.blocks[0]
    n_kv_total = cfg.n_kv_heads * (cfg.d_model // cfg.n_heads)
    _check(
        "GQA: k_proj out_features uses n_kv_heads=2",
        block.attn.k_proj.out_features == n_kv_total,
        f"got {block.attn.k_proj.out_features}",
    )


def test_gqa_has_fewer_kv_params_than_mha() -> None:
    """GQA strictly reduces KV projection size vs MHA at same n_heads."""
    torch.manual_seed(0)
    m_mha = TransformerLM(_make_cfg(n_heads=4, n_kv_heads=4))
    torch.manual_seed(0)
    m_gqa = TransformerLM(_make_cfg(n_heads=4, n_kv_heads=2))
    p_mha = count_params(m_mha)
    p_gqa = count_params(m_gqa)
    _check(
        "GQA has fewer total params than MHA (KV projection shrunk)",
        p_gqa < p_mha,
        f"MHA={p_mha} GQA={p_gqa} diff={p_mha-p_gqa}",
    )


def test_gqa_forward_output_shape_matches_mha() -> None:
    """GQA and MHA produce identically-shaped outputs at the same d_model / n_heads."""
    torch.manual_seed(0)
    m_mha = TransformerLM(_make_cfg(n_heads=4, n_kv_heads=4)).eval()
    torch.manual_seed(1)  # different seed so weights differ
    m_gqa = TransformerLM(_make_cfg(n_heads=4, n_kv_heads=2)).eval()
    ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        y_mha, _ = m_mha(ids)
        y_gqa, _ = m_gqa(ids)
    _check(
        "MHA / GQA outputs have same shape",
        y_mha.shape == y_gqa.shape == (2, 8, 128),
    )


def test_kv_cache_smaller_for_gqa() -> None:
    """KV cache at the same seq_len uses less memory for GQA than MHA."""
    cfg_mha = _make_cfg(n_heads=4, n_kv_heads=4)
    cfg_gqa = _make_cfg(n_heads=4, n_kv_heads=2)
    attn_mha = Attention(
        d_model=cfg_mha.d_model,
        n_heads=cfg_mha.n_heads,
        n_kv_heads=cfg_mha.n_kv_heads,
    )
    attn_gqa = Attention(
        d_model=cfg_gqa.d_model,
        n_heads=cfg_gqa.n_heads,
        n_kv_heads=cfg_gqa.n_kv_heads,
    )
    cache_mha = KVCache()
    cache_gqa = KVCache()
    # Simulate a forward pass with B=2, T=64, head_dim=16 (computed by Attention)
    head_dim = cfg_mha.d_model // cfg_mha.n_heads
    B, T = 2, 64
    k_mha = torch.randn(B, cfg_mha.n_kv_heads, T, head_dim)
    v_mha = torch.randn(B, cfg_mha.n_kv_heads, T, head_dim)
    k_gqa = torch.randn(B, cfg_gqa.n_kv_heads, T, head_dim)
    v_gqa = torch.randn(B, cfg_gqa.n_kv_heads, T, head_dim)
    cache_mha.update(k_mha, v_mha)
    cache_gqa.update(k_gqa, v_gqa)
    size_mha = cache_mha.K.numel() + cache_mha.V.numel()
    size_gqa = cache_gqa.K.numel() + cache_gqa.V.numel()
    _check(
        "GQA KV cache is smaller than MHA at the same seq_len",
        size_gqa < size_mha,
        f"MHA={size_mha} GQA={size_gqa} ratio={size_gqa/size_mha:.2f}",
    )


def test_unknown_norm_kind_raises() -> None:
    """Passing an unknown norm_kind must raise ValueError."""
    try:
        _make_cfg(norm_kind="batchnorm")
        TransformerLM(_make_cfg(norm_kind="batchnorm"))
    except ValueError as e:
        _check("unknown norm_kind raises ValueError", "norm_kind" in str(e), str(e))
        return
    _check("unknown norm_kind raises ValueError", False, "no exception raised")


# ----- main ----------------------------------------------------------

def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            _check(f"{t.__name__} (assertion)", False, str(e))
        except Exception as e:  # noqa: BLE001
            _check(f"{t.__name__} (exception)", False, f"{type(e).__name__}: {e}")

    failed = [r for r in _results if not r[1]]
    total = len(_results)
    print()
    print(f"=== {total - len(failed)} / {total} passed ===")
    if failed:
        print("FAILED:")
        for name, _, msg in failed:
            print(f"  {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())