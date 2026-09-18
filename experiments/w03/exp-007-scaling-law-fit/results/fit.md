# W3 — Scaling-law fit + hold-out prediction

Fit the Chinchilla-style power law on the 4 non-hold-out points, then predict the held-out point (the largest model, 20M-class).

## Pilot summary

| Label | N (params) | Val loss |
| --- | ---: | ---: |
| `1m` | 1,642,496 | 4.1495 |
| `2_5m` | 3,590,400 | 3.7409 |
| `5m` | 6,595,584 | 3.4538 |
| `10m` | 14,891,008 | 3.0527 |

## Fit

Model: `L(N) = L_inf + a · N^(-α)`. Fitted by 1-D grid search over `L_inf` with closed-form least squares for `a` and `α`.

- **L_inf** = `1.5264`
- **a**     = `88.0205`
- **α**     = `0.2446` (exponent on N)

## Hold-out prediction

**Hold-out point (20M) not yet trained.** Re-run `experiments/w03/exp-007-scaling-law-fit/runner.py` after `exp-006-scaling-pilot-20m` finishes to get the extrapolation error.

## What this means

Fit is computed on the 4 small points; the 20M hold-out comparison will be filled in when that pilot completes.

## Caveats

- **Iso-data, not iso-compute**: each pilot trained on the same 
  ~386K-token corpus. This isolates the model-size effect, but 
  real scaling laws (Hoffmann 2022) optimize compute jointly.
- **100-step budget**: convergence is far below what production 
  sweeps use. The `L(N)` curve here describes *partially-trained* 
  loss, not converged loss.
- **Tiny corpus (1 MB)**: extrapolation outside this regime is 
  not informative.

## Files

- [`results/fit.json`](./results/fit.json)
- [`results/scaling_curve.png`](./results/scaling_curve.png)
- [`runner.py`](./runner.py)