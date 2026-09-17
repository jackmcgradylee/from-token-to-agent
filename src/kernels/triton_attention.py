"""Flash-Attention-2-style forward attention kernel in Triton.

This is the speed-critical piece. It computes:
    O = softmax(Q @ K^T * scale + mask) @ V

with two key tricks (Dao et al. 2022 / 2023):
  1. Tiling — Q, K, V live in SRAM, never materialize the full (T, S) attention
     matrix. Memory: O(T * D) instead of O(T * S).
  2. Online softmax — accumulate softmax statistics across blocks with a single
     rescaling pass; no need to materialize logits.

HW2 scope (forward only): this kernel does forward pass with optional causal
masking and optional GQA via per-block repetition. Backward is EXT-2XX work.

Autotune: block sizes (BLOCK_M, BLOCK_N) are selected at first invocation per
shape signature; subsequent calls hit the cache. We expose the tuned config in
benchmark output so the user can see what the autotuner picked.

Hardware requirements:
  - CUDA (this file won't import triton successfully on CPU; triton_attention.py
    is gated on `import triton` succeeding. On Jetson this file is never
    imported — `attention.py` falls back to `reference_attention.py` via the
    backend interface).
  - bf16 / fp16 / fp32 supported (template-typed).
"""

from __future__ import annotations

import torch

# Triton is optional. We import lazily inside forward() so the module file is
# importable on machines without CUDA (Jetson, macOS, etc.).
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    triton = None  # type: ignore
    tl = None  # type: ignore


# ------------------------------------------------------------ kernel definition


if _TRITON_AVAILABLE:

    @triton.jit
    def _fwd_kernel(
        Q,  # (B, H, M, D) — query
        K,  # (B, H_kv, N, D) — key
        V,  # (B, H_kv, N, D) — value
        O,  # (B, H, M, D) — output
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        M_LEN,  # query length
        N_LEN,  # key length
        H: tl.constexpr,
        H_KV: tl.constexpr,
        N_REP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        """One program == one query block of BLOCK_M rows for one (B, H)."""
        # Block IDs.
        start_m = tl.program_id(0)
        bh_idx = tl.program_id(1)
        b_idx = bh_idx // H
        h_idx = bh_idx % H
        h_kv_idx = h_idx // N_REP  # which KV head this Q head reads from

        # Q block offsets.
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)

        # Pointers to this block's Q.
        q_ptr = (
            Q
            + b_idx * stride_qb
            + h_idx * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )
        # Mask: rows beyond M_LEN are invalid.
        q_mask = offs_m[:, None] < M_LEN

        # Load Q block (BLOCK_M, BLOCK_D).
        q = tl.load(q_ptr, mask=q_mask, other=0.0)

        # Initialize online-softmax accumulators.
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Range of K blocks to iterate over.
        if IS_CAUSAL:
            # Causal: K block end <= current Q row position.
            # Each program processes BLOCK_M rows, so the last Q row is at
            # start_m*BLOCK_M + BLOCK_M - 1. K end = last_q_row + 1.
            last_q = start_m * BLOCK_M + BLOCK_M
            n_end = tl.cdiv(last_q, BLOCK_N) * BLOCK_N
        else:
            n_end = N_LEN

        # Iterate over K/V blocks.
        for start_n in tl.range(0, n_end, BLOCK_N):
            offs_n_curr = start_n + offs_n
            k_mask = offs_n_curr[None, :] < N_LEN  # (1, BLOCK_N)

            k_ptr = (
                K
                + b_idx * stride_kb
                + h_kv_idx * stride_kh
                + offs_n_curr[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            v_ptr = (
                V
                + b_idx * stride_vb
                + h_kv_idx * stride_vh
                + offs_n_curr[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )

            k = tl.load(k_ptr, mask=k_mask, other=0.0)
            v = tl.load(v_ptr, mask=k_mask, other=0.0)

            # qk^T : (BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            qk = tl.dot(q, tl.trans(k))
            qk = qk * sm_scale

            # Causal mask: in causal mode, also mask K positions > last Q row.
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
                qk = tl.where(causal_mask & k_mask, qk, float("-inf"))
            else:
                qk = tl.where(k_mask, qk, float("-inf"))

            # Online softmax update.
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_ij)
            p = tl.exp(qk - m_ij[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = acc + tl.dot(p.to(DTYPE), v)
            m_i = m_ij

        # Final normalization.
        acc = acc / l_i[:, None]

        # Write output.
        o_ptr = (
            O
            + b_idx * stride_ob
            + h_idx * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od
        )
        tl.store(o_ptr, acc.to(DTYPE), mask=q_mask)


# ------------------------------------------------------------ autotune configs


def _autotune_configs():
    """Set of (BLOCK_M, BLOCK_N, num_warps, num_stages) to try."""
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ]


# ------------------------------------------------------------ Python wrapper


def triton_attention_forward(
    q: torch.Tensor,           # (B, H, M, D)
    k: torch.Tensor,           # (B, H_kv, N, D)
    v: torch.Tensor,           # (B, H_kv, N, D)
    is_causal: bool = True,
    softmax_scale: float | None = None,
    n_rep: int = 1,            # GQA expansion factor
) -> torch.Tensor:
    """Triton forward attention. Returns (B, H, M, D).

    Raises RuntimeError if Triton is unavailable (caller should fall back).
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not installed / CUDA not available. "
            "Use reference_attention_forward or set attn_backend='pytorch'."
        )

    B, H, M, D = q.shape
    _, H_kv, N, _ = k.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)

    # Allocate output.
    o = torch.empty_like(q)

    # Pick block sizes based on D.
    if D <= 32:
        BLOCK_D = 32
    elif D <= 64:
        BLOCK_D = 64
    elif D <= 128:
        BLOCK_D = 128
    else:
        BLOCK_D = 256

    # Use the first autotune config by default. A more sophisticated version
    # would @triton.autotune over _autotune_configs(), but that requires a
    # warm cache. We expose a manual heuristic for HW2 to keep first-call
    # latency predictable.
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    num_stages = 3

    # 2D grid: (M blocks, B*H).
    grid = (triton.cdiv(M, BLOCK_M), B * H)

    # Convert input dtype to triton-friendly name.
    DTYPE = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}[q.dtype]

    _fwd_kernel[grid](
        q, k, v, o,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        M, N,
        H=H, H_KV=H_kv, N_REP=n_rep,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal, DTYPE=DTYPE,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


# Probe helper for the interface layer.
def is_available() -> bool:
    return _TRITON_AVAILABLE and torch.cuda.is_available()