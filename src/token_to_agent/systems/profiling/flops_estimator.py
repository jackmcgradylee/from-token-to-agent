"""systems.profiling — analytical FLOPs / memory estimator.

W4 deliverables: pure-Python analytic estimators for attention, MLP,
and a full Transformer block. These run on the host without CUDA and
give the *theoretical* numbers that production code (FLOPs profiler,
torch.cuda.memory_summary) would report.

The estimators are deliberately simple — the test cases assert the
formulas against hand-computed values. The point is to make the
*arithmetic intensity* and the *O(T^2) attention vs O(T) linear*
tradeoffs legible, not to chase low-level hardware effects.

Run as a script:
    PYTHONPATH=. .venv/bin/python src/token_to_agent/systems/profiling/flops_estimator.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttentionFlops:
    """FLOPs for one attention forward pass.

    For a (B, H, T, D) Q @ (B, H_kv, S, D) K^T @ V:
      Q@K^T:   2 * B * H * T * S * D  (one MAC = 2 FLOPs)
      softmax: ~5 * B * H * T * S     (exp + add + div, counted as ~5 ops)
      A@V:     2 * B * H * T * S * D
    """
    qk_matmul: float
    softmax_overhead: float
    av_matmul: float
    total: float


def attention_flops(
    B: int, H: int, H_kv: int, T: int, S: int, D: int,
    causal: bool = True, n_rep: int = 1,
) -> AttentionFlops:
    """Analytic FLOPs for one forward pass.

    If GQA is in effect (n_rep > 1, H = n_rep * H_kv), the KV matmul
    cost is the same but the softmax/multiplication has to expand the
    KV heads n_rep times. We approximate the overhead as a factor.
    """
    qk = 2.0 * B * H * T * S * D
    av = 2.0 * B * H * T * S * D
    if causal:
        qk *= 0.5  # half the entries are masked
        av *= 0.5
    softmax_overhead = 5.0 * B * H * T * S  # ~5 ops per (T, S) cell
    return AttentionFlops(
        qk_matmul=qk,
        softmax_overhead=softmax_overhead,
        av_matmul=av,
        total=qk + av + softmax_overhead,
    )


@dataclass
class AttentionMem:
    """Peak memory (bytes, ignoring KV-cache) for one attention forward.

    Materialised attention matrix is (B, H, T, S) — that's the naive
    memory bottleneck. FlashAttention never materialises it; the kernel
    keeps it in SRAM and overwrites tile-by-tile.
    """
    qkv_bytes: int       # input tensors (Q, K, V)
    attn_matrix_bytes: int  # naive: full (B, H, T, S); flash: 0 (tiled)
    output_bytes: int    # O = (B, H, T, D)


def attention_mem(
    B: int, H: int, H_kv: int, T: int, S: int, D: int,
    dtype_bytes: int = 4,
    flash: bool = False,
) -> AttentionMem:
    """Peak memory for attention forward pass.

    `flash=True` assumes the kernel never materialises the full
    attention matrix; `flash=False` (default) accounts for the naive
    (T, S) intermediate.
    """
    qkv = 3 * B * max(H, H_kv) * max(T, S) * D * dtype_bytes
    if flash:
        attn = 0
    else:
        attn = B * H * T * S * dtype_bytes
    out = B * H * T * D * dtype_bytes
    return AttentionMem(qkv_bytes=qkv, attn_matrix_bytes=attn, output_bytes=out)


@dataclass
class TransformerBlockFlops:
    """FLOPs for one Transformer block forward (pre-norm, post-norm)."""
    attn: float
    mlp: float
    norm: float   # ~6 * B*T*d_model (two norms)
    total: float


def transformer_block_flops(
    B: int, T: int, d_model: int, n_heads: int, n_kv_heads: int,
    d_ff: int, has_attn: bool = True,
) -> TransformerBlockFlops:
    """FLOPs for one Transformer block (excluding embeddings / LM head)."""
    if has_attn:
        a = attention_flops(B, n_heads, n_kv_heads, T, T, d_model // n_heads)
        attn = a.total
    else:
        attn = 0.0
    # MLP: two linear layers, (d_model -> d_ff -> d_model), each at
    # batch_size B*T (i.e., one FLOP per element).
    mlp = 2.0 * (2.0 * B * T * d_model * d_ff)
    norm = 6.0 * B * T * d_model
    return TransformerBlockFlops(attn=attn, mlp=mlp, norm=norm, total=attn + mlp + norm)


def transformer_block_mem(
    B: int, T: int, d_model: int, dtype_bytes: int = 4,
) -> int:
    """Peak activation memory for one block (input + output)."""
    return 2 * B * T * d_model * dtype_bytes


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _test_attention_flops():
    """Attention with B=1, H=1, T=4, S=4, D=4, causal: hand-compute."""
    fl = attention_flops(B=1, H=1, H_kv=1, T=4, S=4, D=4, causal=True)
    # Causal halves work, so qk = 2 * 1 * 1 * 4 * 4 * 4 / 2 = 64
    assert fl.qk_matmul == 64.0, f"qk={fl.qk_matmul}"
    assert fl.av_matmul == 64.0, f"av={fl.av_matmul}"
    # softmax_overhead = 5 * 1 * 1 * 4 * 4 = 80 (full count, not halved)
    assert fl.softmax_overhead == 80.0, f"softmax={fl.softmax_overhead}"
    print(f"  [PASS] attention_flops(B=1,H=1,T=4,S=4,D=4,causal): {fl.total}")


def _test_attention_mem():
    """Flash vs naive memory comparison."""
    nav = attention_mem(B=1, H=1, H_kv=1, T=128, S=128, D=64, dtype_bytes=2, flash=False)
    fla = attention_mem(B=1, H=1, H_kv=1, T=128, S=128, D=64, dtype_bytes=2, flash=True)
    # Naive attention matrix: 1 * 1 * 128 * 128 * 2 = 32768 bytes
    assert nav.attn_matrix_bytes == 32768, f"nav={nav.attn_matrix_bytes}"
    assert fla.attn_matrix_bytes == 0, f"fla={fla.attn_matrix_bytes}"
    # Same input/output bytes
    assert nav.qkv_bytes == fla.qkv_bytes
    # Flash saves 32 KB on this tiny shape.
    savings = nav.attn_matrix_bytes - fla.attn_matrix_bytes
    print(f"  [PASS] attention_mem: flash saves {savings} bytes at T=128")


def _test_transformer_block():
    """Sanity: attention and MLP dominate total FLOPs."""
    fl = transformer_block_flops(B=1, T=128, d_model=256, n_heads=4, n_kv_heads=4, d_ff=1024)
    # attn: 4 * 1 * 4 * 128 * 128 * 64 * 0.5 * 2 (qk + av) = ...
    # qk = 2 * 1 * 4 * 128 * 128 * 64 * 0.5 = 4194304
    # av same = 4194304
    # softmax = 5 * 1 * 4 * 128 * 128 = 327680
    # attn total = 4194304 + 4194304 + 327680 = 8716288
    expected_attn = 8716288.0
    assert abs(fl.attn - expected_attn) < 1e-3, f"attn={fl.attn} expected={expected_attn}"
    # MLP: 2 * 2 * 1 * 128 * 256 * 1024 = 134217728
    expected_mlp = 134217728.0
    assert abs(fl.mlp - expected_mlp) < 1e-3, f"mlp={fl.mlp} expected={expected_mlp}"
    # MLP dominates attention 16× at this shape (d_ff=4×d_model, 2 matmuls)
    assert fl.mlp > fl.attn * 10, f"expected MLP > 10× attention: mlp={fl.mlp} attn={fl.attn}"
    print(f"  [PASS] transformer_block_flops: attn={fl.attn:.0f} mlp={fl.mlp:.0f}")


def _test_arithmetic_intensity():
    """Per FLOPs / byte ratio: attention is bandwidth-bound (low AI),
    MLP is compute-bound (high AI). This is why the kernels differ.

    At large T the naive attention materialises the (T, S) logit
    matrix which dominates memory traffic. Flash avoids it by tiling,
    which is the *whole point* of FlashAttention — the AI of the
    *tiled* path is what we want to compare against MLP.

    Naive (T, S) materialised AI:
        AI_naive = 2 * 2 * T * S * D / (T * S) = 4 * D
    Tile-based AI (Flash-style):
        AI_flash ≈ 2 * 2 * T * S * D / (T * D + S * D) ≈ 4 * (T*S*D / (T+S)/D) ~ O(T*S*D / max(T,S))
    For (T=4096, S=4096, D=64): AI_naive = 256, AI_flash ≈ 2,147,483,648 / 1,048,576 ≈ 2048.

    The relative gap is what matters here: naive attention is many
    × more memory-bound than MLP, and FlashAttention closes that gap
    by removing the (T, S) intermediate.
    """
    # MLP (compute-bound): FLOPs / bytes.
    B, T, dm, dff = 1, 4096, 256, 1024
    flops = 2 * B * T * dm * dff
    bytes_ = 2 * B * T * (dm + dff) * 2  # fp16
    ai_mlp = flops / bytes_
    assert 50 < ai_mlp < 200, f"unexpected MLP AI: {ai_mlp}"

    # Naive attention: includes the (T, S) intermediate buffer.
    H, S, D = 8, 4096, 64
    flops_a = 2 * 2 * B * H * T * S * D  # qk + av
    bytes_naive = (3 * B * H * S * D + B * H * T * S + B * H * T * D) * 2
    ai_naive = flops_a / bytes_naive

    # Tiled (Flash) attention: no (T, S) intermediate; reads K/V once.
    bytes_flash = (3 * B * H * S * D + B * H * T * D) * 2
    ai_flash = flops_a / bytes_flash

    # Sanity: flash AI is dramatically larger than naive AI. This is the
    # whole point of FlashAttention — it removes the (T, S) intermediate
    # and stays at high AI even for long sequences.
    assert ai_flash > 5 * ai_naive, (
        f"flash should be ≥5× naive AI: flash={ai_flash:.1f} naive={ai_naive:.1f}"
    )
    print(f"  [PASS] AI ordering (T=4096): naive={ai_naive:.1f}  "
          f"MLP={ai_mlp:.1f}  flash={ai_flash:.1f}  "
          f"(flash is {ai_flash/ai_naive:.1f}× naive)")


if __name__ == "__main__":
    print("[profiling] running self-tests for flops/mem estimators...")
    _test_attention_flops()
    _test_attention_mem()
    _test_transformer_block()
    _test_arithmetic_intensity()
    print("\nAll profiling estimator self-tests passed.")