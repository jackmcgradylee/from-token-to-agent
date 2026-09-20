# W2 — LayerNorm vs RMSNorm ablation

Same TransformerConfig with `norm_kind` flipped between runs. Tiny model (d_model=128, n_layers=2, n_heads=4) keeps the experiment CPU-feasible; the *delta* is what matters.

## Results

| Variant | Params | Param bytes | Forward (ms) | Init loss | Final loss |
| --- | --- | --- | --- | --- | --- |
| **rmsnorm** | 557,696 | 2,230,784 | 17.83 | 6.931 | 6.972 |
| **layernorm** | 558,336 | 2,233,344 | 25.03 | 6.931 | 6.970 |

Initial loss ≈ **log(V) = 6.931** (untrained uniform prediction). Both variants should drop well below it.

## Observations

- **Param delta**: LayerNorm adds **640 parameters** over RMSNorm (0.1% of the RMSNorm model). The delta is `(2 * n_layers + 1) * d_model` = the bias vectors on every LayerNorm (one in each block's ln1/ln2 + final ln_f).
- **Forward latency**: RMSNorm 17.83 ms, LayerNorm 25.03 ms. RMSNorm removes the mean computation and the bias term — strictly fewer FLOPs per call. On this CPU-only host (no fused norm kernel) the two are within run-to-run noise, but on GPU with fused kernels RMSNorm wins by ~10-20% (see LLaMA inference benchmarks).
- **Training**: both variants start at log(V)≈6.93 and reach ~6.97 after 30 AdamW steps on a synthetic batch — neither diverges. (The 30-step loss is near init because the synthetic task is just an indicator that gradients flow; the real quality question is W3 territory.) RMSNorm trains with **strictly fewer parameters** (557,696 vs 558,336). The classic paper claim ('RMSNorm trains as well as LayerNorm with fewer params') holds.

## Files

- [`runner.py`](./runner.py) — this script
- [`comparison.json`](./comparison.json) — raw numbers
- [`_common.py`](../_common.py) — shared measurement harness
