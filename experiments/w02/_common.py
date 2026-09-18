"""W2 ablation runners: LayerNorm vs RMSNorm, MHA vs GQA.

Two experiments share the same scaffolding; this module is the
common harness. Each experiment has its own CLI entry-point script
under experiments/w02/exp-*/runner.py that calls run_norm_ablation()
or run_attention_ablation() with a few hyperparameters.

Measured quantities:
  - Total parameter count
  - Bytes for parameter memory (fp32)
  - KV cache bytes per layer at a fixed sequence length
  - Forward latency (CPU, fp32, n_iters averaged)
  - Validation loss after N tiny training steps (proxies
    "trains at all" rather than quality)

Both ablations are deliberately tiny so they run on CPU in <30s
each. The point is the *delta*, not the absolute numbers.
"""

from __future__ import annotations

import statistics as stats
import time
from dataclasses import dataclass
from typing import Iterable

import torch

from src.token_to_agent.from_scratch.model import (
    KVCache,
    TransformerConfig,
    TransformerLM,
    count_params,
)


# --- Shared measurement helpers ---------------------------------------


def _count_kv_cache_bytes(cfg: TransformerConfig, batch_size: int, seq_len: int) -> int:
    """Memory cost of a single layer's KV cache at given batch + seq."""
    # Each token holds K and V tensors of shape (B, n_kv_heads, head_dim).
    head_dim = cfg.d_model // cfg.n_heads
    elements_per_layer = 2 * batch_size * cfg.n_kv_heads * seq_len * head_dim
    return elements_per_layer * 4  # fp32


def _forward_latency_ms(
    model: TransformerLM,
    input_ids: torch.Tensor,
    n_iters: int,
    warmup: int = 2,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)
        samples = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model(input_ids)
            samples.append((time.perf_counter() - t0) * 1000)
    return stats.mean(samples)


def _tiny_train_loss(
    model: TransformerLM,
    cfg: TransformerConfig,
    n_steps: int = 20,
    batch_size: int = 4,
    seq_len: int = 64,
    lr: float = 1e-3,
) -> float:
    """Run a few AdamW steps on a synthetic batch to verify 'trains at all'.

    Returns the final loss. We don't expect the loss to converge —
    we just want both variants to be trainable.
    """
    torch.manual_seed(0)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(n_steps):
        ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        _, loss = model(ids, targets=targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return float(loss.item())


# --- Norm ablation ---------------------------------------------------


@dataclass
class NormVariantResult:
    norm_kind: str
    n_params: int
    param_bytes: int
    forward_latency_ms: float
    final_train_loss: float
    trained_from: float  # initial loss


def run_norm_ablation(
    norm_kind: str,
    cfg: TransformerConfig,
    n_iters: int = 5,
    train_steps: int = 30,
) -> NormVariantResult:
    torch.manual_seed(0)
    cfg_norm = TransformerConfig(**{**cfg.__dict__, "norm_kind": norm_kind})
    model = TransformerLM(cfg_norm)
    n_params = count_params(model)
    param_bytes = n_params * 4

    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    fwd_ms = _forward_latency_ms(model, input_ids, n_iters=n_iters)

    # Measure trainability.
    torch.manual_seed(0)
    model_init = TransformerLM(TransformerConfig(**{**cfg.__dict__, "norm_kind": norm_kind}))
    torch.manual_seed(0)
    model_train = TransformerLM(TransformerConfig(**{**cfg.__dict__, "norm_kind": norm_kind}))
    # Snapshot the initial loss for delta reporting.
    torch.manual_seed(0)
    init_loss = _tiny_train_loss(model_init, cfg_norm, n_steps=1)
    final_loss = _tiny_train_loss(model_train, cfg_norm, n_steps=train_steps)

    return NormVariantResult(
        norm_kind=norm_kind,
        n_params=n_params,
        param_bytes=param_bytes,
        forward_latency_ms=fwd_ms,
        final_train_loss=final_loss,
        trained_from=init_loss,
    )


# --- Attention ablation ----------------------------------------------


@dataclass
class AttnVariantResult:
    n_heads: int
    n_kv_heads: int
    label: str  # "MHA" or "GQA-4"
    n_params: int
    param_bytes: int
    kv_cache_bytes_per_layer: int
    forward_latency_ms: float
    final_train_loss: float


def run_attention_ablation(
    n_heads: int,
    n_kv_heads: int,
    cfg: TransformerConfig,
    n_iters: int = 5,
    train_steps: int = 30,
) -> AttnVariantResult:
    torch.manual_seed(0)
    cfg_attn = TransformerConfig(**{**cfg.__dict__, "n_heads": n_heads, "n_kv_heads": n_kv_heads})
    model = TransformerLM(cfg_attn)
    n_params = count_params(model)
    param_bytes = n_params * 4

    kv_bytes = _count_kv_cache_bytes(cfg_attn, batch_size=4, seq_len=512)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    fwd_ms = _forward_latency_ms(model, input_ids, n_iters=n_iters)

    torch.manual_seed(0)
    model_train = TransformerLM(TransformerConfig(**{**cfg.__dict__, "n_heads": n_heads, "n_kv_heads": n_kv_heads}))
    final_loss = _tiny_train_loss(model_train, cfg_attn, n_steps=train_steps)

    label = "MHA" if n_kv_heads == n_heads else f"GQA-{n_kv_heads}"
    return AttnVariantResult(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        label=label,
        n_params=n_params,
        param_bytes=param_bytes,
        kv_cache_bytes_per_layer=kv_bytes,
        forward_latency_ms=fwd_ms,
        final_train_loss=final_loss,
    )