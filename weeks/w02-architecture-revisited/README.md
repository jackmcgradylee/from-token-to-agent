# W2 — Architecture Revisited

## What this week is about

What remains of the original 2017 Transformer in a modern decoder-only LLM, and which architectural choices materially affect training and inference?

## Status

In Progress. The baseline decoder-only Transformer is implemented; the W2 ablation runs are not yet written.

## Implemented so far

- `src/token_to_agent/model/model.py` — `TransformerConfig` + `TransformerLM` + `TransformerBlock`
  - Pre-norm residual blocks
  - Tied LM head
  - Truncated-normal init (`std=0.02`)
  - KV cache (per layer, `KVCache` class)
  - `forward(x, targets=None, kv_cache=None, offset=0)` and `generate(...)` (with and without KV cache)
- `src/token_to_agent/model/rmsnorm.py` — `RMSNorm` (fp32 RMS for stability under bf16)
- `src/token_to_agent/model/rope.py` — `precompute_rope_cache(...)` + `apply_rope(...)`
- `src/token_to_agent/model/swiglu.py` — `SwiGLU` (fused gate + up projection)
- `src/token_to_agent/model/attention.py` — `Attention` with GQA (`n_kv_heads <= n_heads`)
- `src/token_to_agent/model/utils.py` — `count_params(model, by_component=True)`
- `tests/model/test_transformer.py` — 4 tests, all passing: forward+backward, KV-cache generation, RoPE position dependence, parameter count breakdown.

## Architecture ablations (W2 deliverables — pending)

- [ ] **RMSNorm vs LayerNorm** — `src/token_to_agent/model/layernorm.py` not yet written.
- [ ] **MHA vs GQA** — same code path, just toggle `n_kv_heads` between `n_heads` and `n_heads/2` etc. — needs an ablation runner.
- [ ] **Parameter / FLOPs calculator** — count is implemented; FLOPs/token estimate is not.
- [ ] Two ablation reports under `experiments/w02/`.

## Pass criteria (from COURSE.md §2 W2)

- [ ] forward/backward tests pass. ✅
- [ ] model can overfit a tiny batch. (manual check; needs a script)
- [ ] model can generate autoregressively. ✅ (test_kv_cache_generation)
- [ ] architectural ablations use matched training budgets.

## Open questions

- How do I implement LayerNorm without breaking the current architecture code's pre-norm pattern? (Probably a sibling RMSNorm-shaped module.)
- For the FLOPs/token estimator: count matmul + attn FLOPs, ignore elementwise? Or include everything for honesty?
- For the overfit-a-tiny-batch check: should it be `assert loss < 0.5 after N steps on a 32-example batch`?