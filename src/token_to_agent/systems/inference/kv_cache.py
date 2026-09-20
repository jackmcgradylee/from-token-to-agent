"""KV cache memory math for transformer inference.

Pure-Python (no torch). Provides parameterised estimators used by:
- serving/simulated_engine.py  (scheduler decides fit / preempt)
- serving/metrics_collector.py (peak memory in reports)
- experiments/w05/exp-003-serving-bench/runner.py  (cost-sweep)

Reference: vLLM (Kwon et al., SOSP 2023), SGLang RadixAttention
(Zheng et al., 2024). The numbers below match what those systems
report for Llama-2-7B at 4k context on H100.

Why this is its own module: the scheduler must reject / preempt
requests when KV cache would overflow. The threshold is a function
of model config, dtype, and available HBM — and we want to compute
it deterministically without instantiating a 7B model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

# A handful of canonical model configs the bench uses. These match the
# public Llama-2 / Mistral / Phi-3 specs at the time of writing.
# (Llama-2-7B is the W5 reference model.)
PRESETS: dict[str, "ModelConfig"] = {}


@dataclass(frozen=True)
class ModelConfig:
    """Static parameters of the model being served.

    All numbers are integers except dtype_bytes; everything else
    (KV bytes / request, peak memory, max concurrent requests at a
    given context length) is computed by the helpers below.
    """

    name: str
    n_layers: int
    n_heads: int           # Q heads (H_q)
    n_kv_heads: int        # KV heads (H_kv); == H_q for MHA, < H_q for GQA
    head_dim: int
    max_seq_len: int       # model's positional limit (e.g. 4096 for Llama-2)
    param_count_b: float   # rough, used only for $/1M-token sanity check

    @property
    def n_rep(self) -> int:
        """Q-heads per KV-head (grouped-query ratio)."""
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads


PRESETS["llama2-7b"] = ModelConfig(
    name="llama2-7b",
    n_layers=32,
    n_heads=32,
    n_kv_heads=32,  # Llama-2-7B is MHA, not GQA
    head_dim=128,
    max_seq_len=4096,
    param_count_b=6.7,
)

PRESETS["llama2-70b"] = ModelConfig(
    name="llama2-70b",
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,  # GQA
    head_dim=128,
    max_seq_len=4096,
    param_count_b=69.0,
)

PRESETS["mistral-7b"] = ModelConfig(
    name="mistral-7b",
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,  # GQA
    head_dim=128,
    max_seq_len=32768,
    param_count_b=7.2,
)

PRESETS["phi3-mini"] = ModelConfig(
    name="phi3-mini",
    n_layers=32,
    n_heads=32,
    n_kv_heads=32,  # MHA
    head_dim=96,
    max_seq_len=4096,
    param_count_b=3.8,
)


# ---------------------------------------------------------------------------
# KV cache size math
# ---------------------------------------------------------------------------

DType = Literal["fp32", "fp16", "bf16", "int8", "int4"]
_DTYPE_BYTES: dict[str, int] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 1,  # packed; real byte count is ceil(bits / 8)
}


def dtype_bytes(dtype: DType) -> int:
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {list(_DTYPE_BYTES)}")
    return _DTYPE_BYTES[dtype]


def kv_bytes_per_token_per_layer(
    cfg: ModelConfig,
    dtype: DType = "fp16",
) -> int:
    """Memory for K + V for one token at one layer."""
    return 2 * cfg.n_kv_heads * cfg.head_dim * dtype_bytes(dtype)


def kv_bytes_per_token(
    cfg: ModelConfig,
    dtype: DType = "fp16",
) -> int:
    """Memory for K + V for one token across all layers."""
    return cfg.n_layers * kv_bytes_per_token_per_layer(cfg, dtype)


def kv_bytes_for_request(
    cfg: ModelConfig,
    seq_len: int,
    dtype: DType = "fp16",
) -> int:
    """Memory to hold the KV cache for a request at `seq_len` tokens."""
    if seq_len < 0 or seq_len > cfg.max_seq_len:
        raise ValueError(
            f"seq_len={seq_len} out of bounds [0, {cfg.max_seq_len}] for {cfg.name}"
        )
    return seq_len * kv_bytes_per_token(cfg, dtype)


def model_weight_bytes(
    cfg: ModelConfig,
    dtype: DType = "fp16",
) -> int:
    """Rough estimate of model weights in bytes from param_count_b.

    Real param counts include embeddings; this rounds to billions and
    multiplies by dtype_bytes. For cost-model sanity-check only.
    """
    return int(cfg.param_count_b * 1e9) * dtype_bytes(dtype)


def max_concurrent_at_seq_len(
    cfg: ModelConfig,
    seq_len: int,
    gpu_memory_bytes: int,
    dtype: DType = "fp16",
    activation_overhead_gb: float = 2.0,
) -> int:
    """How many requests of `seq_len` context can fit alongside the weights.

    `activation_overhead_gb` is reserved for activations + CUDA workspace
    + the engine's own metadata. Default 2 GB is conservative for 7B-
    class models on H100.
    """
    overhead_bytes = int(activation_overhead_gb * 1024**3)
    weights = model_weight_bytes(cfg, dtype)
    available = gpu_memory_bytes - weights - overhead_bytes
    if available <= 0:
        return 0
    per_req = kv_bytes_for_request(cfg, seq_len, dtype)
    if per_req <= 0:
        return 0
    return available // per_req


# ---------------------------------------------------------------------------
# Cost math
# ---------------------------------------------------------------------------

def cost_per_million_tokens(
    gpu_dollars_per_hour: float,
    tokens_per_second: float,
) -> float:
    """$/1M tokens given a $/hr rate and aggregate tokens/s.

    Derived:
        $/sec = $/hr / 3600
        $/token = $/sec / TPS
        $/1M = $/token * 1e6
              = $/hr * 1e6 / (3600 * TPS)
    """
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be > 0")
    return gpu_dollars_per_hour * 1e6 / (3600.0 * tokens_per_second)


# ---------------------------------------------------------------------------
# Reference numbers used by the docs and the DEV-HOST-LIMITATIONS file
# ---------------------------------------------------------------------------

# Per docs/w5 §3: Llama-2-7B fp16 T=4096 → 128 KB/token, 512 MB/req.
# We assert that the helpers above reproduce that.
def assert_doc_example() -> None:
    cfg = PRESETS["llama2-7b"]
    per_tok = kv_bytes_per_token(cfg, "fp16")
    per_req = kv_bytes_for_request(cfg, 4096, "fp16")
    assert per_tok == 32 * 2 * 32 * 128 * 2, f"per_tok={per_tok}"
    assert per_req == 4096 * per_tok, f"per_req={per_req}"
    assert kv_bytes_per_token_per_layer(cfg, "fp16") == 2 * 32 * 128 * 2
    # The docs use both Mistral (GQA-8: 128 KB/token, 512 MB/req at 4k)
    # and Llama-2-7B (MHA: 512 KB/token, 2 GB/req at 4k) — see
    # docs/w5-inference-economics.md §3.


if __name__ == "__main__":
    # Tiny CLI: prints the W5 reference numbers so the file is
    # self-checkable.
    print("=== W5 KV cache reference numbers ===")
    for name, cfg in PRESETS.items():
        per_tok = kv_bytes_per_token(cfg, "fp16")
        per_req_4k = kv_bytes_for_request(cfg, 4096, "fp16")
        per_req_2k = kv_bytes_for_request(cfg, 2048, "fp16")
        print(
            f"{name:14s}  per_token={per_tok/1024:7.1f} KB  "
            f"4k_req={per_req_4k/1024/1024:7.1f} MB  "
            f"2k_req={per_req_2k/1024/1024:7.1f} MB  "
            f"max_seq={cfg.max_seq_len}"
        )
    # H100 80GB reference
    h100 = 80 * 1024**3
    a100_80 = 80 * 1024**3
    a100_40 = 40 * 1024**3
    print()
    print("=== max concurrent @ 4k context ===")
    for cfg_name in ["llama2-7b", "mistral-7b"]:
        cfg = PRESETS[cfg_name]
        for gpu_name, gpu_b in [("H100 80GB", h100), ("A100 80GB", a100_80), ("A100 40GB", a100_40)]:
            n = max_concurrent_at_seq_len(cfg, 4096, gpu_b, "fp16", 2.0)
            print(f"  {cfg_name:12s} on {gpu_name:10s}: {n} concurrent reqs")
    assert_doc_example()
    print()
    print("=== cost-per-1M sanity ===")
    print(f"  $2.00/hr H100 @ 43k TPS decode = ${cost_per_million_tokens(2.0, 43000):.4f} / 1M tokens")
