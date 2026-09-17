"""Full Transformer LM: token + position embeddings, N x [RMSNorm->Attn->RMSNorm->SwiGLU],
final norm, tied LM head."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, KVCache
from .rmsnorm import RMSNorm
from .rope import precompute_rope_cache
from .swiglu import SwiGLU


@dataclass
class TransformerConfig:
    vocab_size: int = 8192
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None
    head_dim: int | None = None
    d_ff: int | None = None  # default: SwiGLU auto-computed
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-6
    init_std: float = 0.02
    tie_word_embeddings: bool = True


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = Attention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            rope_theta=cfg.rope_theta,
            dropout=cfg.dropout,
        )
        self.ln2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, hidden_dim=cfg.d_ff)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, kv_cache=kv_cache, offset=offset)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.n_kv_heads is None:
            cfg.n_kv_heads = cfg.n_heads
        if cfg.head_dim is None:
            cfg.head_dim = cfg.d_model // cfg.n_heads

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model, eps=cfg.norm_eps)

        # Tied LM head: share weights with token_emb.
        # (Linear weight = token_emb.weight, with no bias.)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.token_emb.weight

        # Precompute RoPE cache as a buffer (moved with the module).
        cos, sin = precompute_rope_cache(
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            theta=cfg.rope_theta,
            device=self.token_emb.weight.device,
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialize weights.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        # Truncated normal with std init_std for Linear and Embedding weights.
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.cfg.init_std, a=-2 * self.cfg.init_std, b=2 * self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=self.cfg.init_std, a=-2 * self.cfg.init_std, b=2 * self.cfg.init_std)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, T)
        targets: torch.Tensor | None = None,  # (B, T) optional
        kv_cache: list[KVCache] | None = None,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_emb(input_ids)

        for i, block in enumerate(self.blocks):
            layer_cache = None if kv_cache is None else kv_cache[i]
            x = block(x, self.rope_cos, self.rope_sin, kv_cache=layer_cache, offset=offset)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,         # (B, T0)
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Naive (non-batched) generation with optional KV cache.

        For a small model this is plenty fast on CPU for HW1 verification.
        HW4 / FINAL can swap in a KV-cached sampler.
        """
        was_training = self.training
        self.eval()

        B, T0 = input_ids.shape
        device = input_ids.device

        if use_cache:
            kv_cache = [KVCache() for _ in range(len(self.blocks))]
            # Prefill: pass full prompt.
            self.forward(input_ids, kv_cache=kv_cache, offset=0)
            cur = input_ids[:, -1:]  # last token id (B, 1)
            offset = T0
            # Cap generation at remaining context budget.
            max_new_tokens = min(max_new_tokens, self.cfg.max_seq_len - T0)
            out_ids = list(input_ids[0].tolist())

            for _ in range(max_new_tokens):
                logits, _ = self.forward(cur, kv_cache=kv_cache, offset=offset - 1)
                next_logits = logits[:, -1, :] / max(temperature, 1e-5)
                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                    next_logits = torch.where(
                        next_logits < v[:, -1:],
                        torch.full_like(next_logits, float("-inf")),
                        next_logits,
                    )
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = torch.cumsum(probs, dim=-1)
                    # Remove tokens with cum > top_p.
                    sorted_mask = cum > top_p
                    # Always keep the first.
                    sorted_mask[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
                    next_logits = torch.zeros_like(next_logits).scatter_(
                        -1, sorted_indices, sorted_logits
                    )
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                out_ids.append(next_id.item())
                cur = next_id
                offset += 1

            if was_training:
                self.train()
            return torch.tensor([out_ids], device=device)
        else:
            # Cache-less slow path: re-encode whole sequence each step.
            ids = input_ids
            out_ids = list(input_ids[0].tolist())
            for _ in range(max_new_tokens):
                logits, _ = self.forward(ids)
                next_logits = logits[:, -1, :] / max(temperature, 1e-5)
                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                    next_logits = torch.where(
                        next_logits < v[:, -1:],
                        torch.full_like(next_logits, float("-inf")),
                        next_logits,
                    )
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                out_ids.append(next_id.item())
                ids = torch.cat([ids, next_id], dim=1)
                if ids.shape[1] > self.cfg.max_seq_len:
                    ids = ids[:, -self.cfg.max_seq_len:]
            if was_training:
                self.train()
            return torch.tensor([out_ids], device=device)