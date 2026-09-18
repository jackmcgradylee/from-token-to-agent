# W4 exp-001 — Attention backend benchmark
Grid: [(1, 4, 4, 64, 32), (1, 4, 4, 128, 32), (1, 8, 2, 256, 64), (1, 8, 2, 512, 64), (2, 8, 2, 1024, 64)]
CUDA available: **False**
| backend | B | H | H_kv | T | D | fwd (ms) | fwd+bwd (ms) | peak mem (MB) | max err | TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive` | 1 | 4 | 4 | 64 | 32 | 1.654 | 5.182 | 0.0 | 0.00e+00 | 0.001 |
| `reference` | 1 | 4 | 4 | 64 | 32 | 1.607 | 5.037 | 0.0 | 0.00e+00 | 0.001 |
| `pytorch` | 1 | 4 | 4 | 64 | 32 | 0.277 | 1.580 | 0.0 | 8.94e-08 | 0.008 |
| `triton` | 1 | 4 | 4 | 64 | 32 | 1.619 | 4.970 | 0.0 | 0.00e+00 | 0.001 |
| `naive` | 1 | 4 | 4 | 128 | 32 | 2.538 | 7.465 | 0.0 | 0.00e+00 | 0.003 |
| `reference` | 1 | 4 | 4 | 128 | 32 | 2.559 | 7.025 | 0.0 | 0.00e+00 | 0.003 |
| `pytorch` | 1 | 4 | 4 | 128 | 32 | 0.873 | 3.265 | 0.0 | 1.19e-07 | 0.010 |
| `triton` | 1 | 4 | 4 | 128 | 32 | 2.632 | 7.265 | 0.0 | 0.00e+00 | 0.003 |
| `naive` | 1 | 8 | 2 | 256 | 64 | 14.566 | 36.300 | 1.9 | 0.00e+00 | 0.009 |
| `reference` | 1 | 8 | 2 | 256 | 64 | 17.608 | 41.109 | 0.2 | 0.00e+00 | 0.008 |
| `pytorch` | 1 | 8 | 2 | 256 | 64 | 9.295 | 58.579 | 0.0 | 1.19e-07 | 0.014 |
| `triton` | 1 | 8 | 2 | 256 | 64 | 15.801 | 37.206 | 2.0 | 0.00e+00 | 0.008 |
| `naive` | 1 | 8 | 2 | 512 | 64 | 62.990 | 107.176 | 0.0 | 0.00e+00 | 0.009 |
| `reference` | 1 | 8 | 2 | 512 | 64 | 46.148 | 107.667 | 0.0 | 0.00e+00 | 0.012 |
| `pytorch` | 1 | 8 | 2 | 512 | 64 | 31.831 | 130.143 | 0.0 | 1.27e-07 | 0.017 |
| `triton` | 1 | 8 | 2 | 512 | 64 | 47.186 | 107.443 | 7.9 | 0.00e+00 | 0.011 |
| `naive` | 2 | 8 | 2 | 1024 | 64 | 315.898 | 894.341 | 3.9 | 0.00e+00 | 0.014 |
| `reference` | 2 | 8 | 2 | 1024 | 64 | 318.176 | 721.150 | 0.2 | 0.00e+00 | 0.013 |
| `pytorch` | 2 | 8 | 2 | 1024 | 64 | 196.276 | 631.320 | 0.0 | 1.19e-07 | 0.022 |
| `triton` | 2 | 8 | 2 | 1024 | 64 | 310.769 | 834.238 | 0.3 | 0.00e+00 | 0.014 |

## Notes
- All times are median of 5 runs after 3 warmup iterations.
- `peak_mem_mb` is process-level RSS delta around the forward pass; for CUDA we'd switch to `torch.cuda.max_memory_allocated`.
- `max_abs_err` is the max absolute difference vs `reference` (fp32).
- On a CUDA-less host, the `triton` row falls back to `reference` via the interface; rerun on a GPU host for a real speedup number.
