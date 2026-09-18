# W4 exp-002 — seq-len sweep
CUDA available: **False**
| backend | T | fwd (ms) | fwd+bwd (ms) | TFLOPS |
| --- | ---: | ---: | ---: | ---: |
| `naive` | 128 | 6.15 | 14.89 | 0.005 |
| `reference` | 128 | 6.35 | 14.86 | 0.005 |
| `pytorch` | 128 | 2.70 | 9.29 | 0.012 |
| `triton` | 128 | 5.53 | 15.02 | 0.006 |
| `naive` | 256 | 15.19 | 36.95 | 0.009 |
| `reference` | 256 | 16.36 | 38.94 | 0.008 |
| `pytorch` | 256 | 8.72 | 28.02 | 0.015 |
| `triton` | 256 | 27.93 | 67.43 | 0.005 |
| `naive` | 512 | 45.43 | 108.22 | 0.012 |
| `reference` | 512 | 44.54 | 106.07 | 0.012 |
| `pytorch` | 512 | 32.81 | 126.32 | 0.016 |
| `triton` | 512 | 41.59 | 100.27 | 0.013 |
| `naive` | 1024 | 212.52 | 499.97 | 0.010 |
| `reference` | 1024 | 242.62 | 356.07 | 0.009 |
| `pytorch` | 1024 | 122.12 | 310.68 | 0.018 |
| `triton` | 1024 | 226.49 | 377.72 | 0.009 |
| `naive` | 2048 | 808.76 | 1581.64 | 0.011 |
| `reference` | 2048 | 784.54 | 1614.21 | 0.011 |
| `pytorch` | 2048 | 324.97 | 1045.21 | 0.026 |
| `triton` | 2048 | 737.00 | 1647.90 | 0.012 |

## Plot
`results/seqlen.png` — log-log latency vs T, both forward and forward+backward.

## Notes
- Naive PyTorch attention has O(T^2) matmul cost; latency should grow roughly as a line with slope 2 on the log-log plot.
- Flash-style tiled kernels achieve O(T) effective bandwidth by never materialising the full (T, T) matrix; the Triton row on a CUDA host should show a flatter curve at large T.
- On a CUDA-less host the Triton row falls back to `reference`; rerun on a GPU host to see the Flash effect.
