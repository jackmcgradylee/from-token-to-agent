# Scaling Laws, From Textbook to First Measurement

> A personal reading of W3 — Training Dynamics & Scaling Laws — turned
> into concrete numbers by the W3 scaling-pilot sweep.

## The question

Transformer models of size N trained on D tokens behave in a remarkably
predictable way. The empirical scaling laws (Hoffmann et al. 2022;
Kaplan et al. 2020; Henighan et al. 2020) all look roughly like:

```text
L(N, D) = L_inf + (N / N_c)^(-alpha) + (D / D_c)^(-beta)
```

Two power laws — one for parameters, one for data — plus an irreducible
loss floor `L_inf`. The textbook claims:

- α ≈ 0.07–0.08 (parameters help)
- β ≈ 0.10–0.35 (data helps, with compute-optimal frontier around
  D ≈ 20N tokens)
- L_inf is model-class-dependent (≈ 1.69 for autoregressive LMs
  in Hoffmann's setup, ≈ 0 for an oracle)

The textbook claims are fit on **billions of parameters** and **trillions
of tokens**, with hundreds of independent runs and proper convergence.
We have a CPU-only the dev host, a 1 MB toy corpus, and 100 training steps
per pilot. None of those numbers matter at our scale. So what *can*
this week's experiment say?

## What this experiment is actually testing

A *first-hand* version of one half of the law: **`L(N) = L_inf + a · N^(-α)`
with D fixed**. Iso-data, not iso-compute. Five runs across one and a
half orders of magnitude in N (1.6M → 33M), same 1 MB corpus, same 100
optimization steps, same lr schedule. The hold-out test is whether the
fit on {1M, 2.5M, 5M, 10M} predicts the 20M run.

## What we already know will limit us

- **100 optimization steps is *not* convergence.** At N=1.6M we land at
  val_loss=4.15 after 100 steps; at N=3.6M we land at 3.74. The "real"
  curve would be steeper because larger models would benefit more from
  additional training. We are measuring *partially-trained loss*.
- **1 MB corpus ≈ 386K tokens.** The Chinchilla frontier is at
  ~20N tokens = ~660M tokens for our biggest model. We have 600× less
  data than the frontier says we need. Larger models will be
  **under-trained**, not **over-parameterized**. The slope α we measure
  here is a *data-limited* slope, not the data-rich one Hoffmann
  reports.
- **Vocabulary is tiny (vocab≈560).** Real scaling laws normalize for
  vocab size; ours doesn't. The embedding table accounts for ~12% of
  total params at N=1.6M and ~3% at N=33M — a slow-moving confounder.

## Setup, concretely

| Pilot | Config | d_model | n_layers | head_dim | actual N |
|---|---|---:|---:|---:|---:|
| 1m | `configs/hw1/scaling-1m.yaml` | 128 | 8 | 32 | **1,642,496** |
| 2_5m | `configs/hw1/scaling-2_5m.yaml` | 160 | 12 | 40 | **3,590,400** |
| 5m | `configs/hw1/scaling-5m.yaml` | 192 | 16 | 48 | **6,887,616** |
| 10m | `configs/hw1/scaling-10m.yaml` | 256 | 20 | 64 | **15,280,384** |
| 20m | `configs/hw1/scaling-20m.yaml` | 320 | 28 | 80 | **33,359,680** |

Every run shares: vocab=2048, BPE merges trained on the 1 MB toy
corpus, seq_len=128, batch=4 × grad_accum=2 (effective batch 8),
lr_peak=3e-4 with linear warmup (20 steps) → cosine to 3e-5,
weight_decay=0.1, grad_clip=1.0, AdamW (β1=0.9, β2=0.95), fp32, seed=42.

Code: [`experiments/w03/_common.py`](../experiments/w03/_common.py) ·
shared harness; five thin `runner.py` files under each pilot directory.

## The hold-out test

Fit `L(N) = L_inf + a · N^(-α)` on {1M, 2.5M, 5M, 10M} by 1-D grid
search over `L_inf` (closed-form OLS for `a` and `α`). Then **predict**
the 20M point and compare against the actually-trained value. That
number — the relative prediction error on a model we never trained in
the fit set — is the only honest way to know whether our 4-point fit
is "real" or "noise from 4 points". Code:
[`experiments/w03/exp-007-scaling-law-fit/runner.py`](../experiments/w03/exp-007-scaling-law-fit/runner.py).

If the relative error is < 10%, the W3 sweep actually behaves like a
power law on this regime — i.e. **even a 4-point fit extrapolates**.
If it is 10-25%, we are in the range where "data-limited α" and
"compute-limited α" start to diverge. If it is > 25%, the curve is
probably not a single power law at this scale and we should fit a
different functional form.

## What this experiment is NOT

- **Not a Chinchilla reproduction.** Hoffmann 2022 trained 400+ models
  to convergence on a 4-trillion-token corpus. We trained 5 models for
  100 steps on a 1 MB corpus. We are testing whether the *functional
  form* of `L(N)` is detectable on a tiny setup, not whether we can
  reproduce Hoffmann's constants.
- **Not a compute-optimal frontier.** `D / N` is wildly out of balance
  here (we have 1 MB of data for everything). The Chinchilla frontier
  would say: at N=33M we need ~660M tokens; we have 0.4M.
- **Not architecture-scaling.** All five pilots use the same RoPE +
  RMSNorm + GQA + SwiGLU stack from HW2. We only vary `d_model` and
  `n_layers`. A proper scaling sweep would also vary `n_heads`,
  `head_dim`, and depth-to-width ratio. Those are extensions (`EXT-W3-0X`).

## What to look at first

1. **Is the slope negative?** i.e. do larger N land at lower loss?
   `experiments/w03/exp-007-scaling-law-fit/results/scaling_curve.png`
   shows this visually.
2. **Does the hold-out prediction land inside the model?** The
   `exp-007/results/fit.md` report prints the predicted vs actual
   numbers and the relative error.
3. **Does the residual of the hold-out point look like a slope or a
   step?** If the 20M point drops a lot *less* than the fit predicts,
   we are seeing **diminishing returns** — under-training. If it drops
   *more*, we are seeing **emergence** — but emergence at this scale
   would be a surprise and worth investigating.

## After W3

The follow-on extensions are queued:

- **EXT-W3-01** — same sweep with `head_dim` and `n_heads` varied
  separately, to decouple "deeper" from "wider".
- **EXT-W3-02** — same sweep on a 10× larger corpus (10 MB) to see if
  α shrinks toward the Chinchilla value as the data-limited regime
  relaxes.
- **EXT-W3-03** — fit `L(N, D)` jointly on an iso-compute grid (vary
  both N and D such that `6ND` is constant) — the actual Chinchilla
  protocol. Defer until W4 (compute work) lands.

## See also

- [`experiments/w03/_common.py`](../experiments/w03/_common.py) — shared pilot harness.
- [`experiments/w03/exp-007-scaling-law-fit/`](../experiments/w03/exp-007-scaling-law-fit/) — fit + hold-out.
- [`weeks/w03-training-dynamics-scaling-laws/README.md`](../weeks/w03-training-dynamics-scaling-laws/README.md) — week-level journal.
- [`docs/signal-ladder.md`](./signal-ladder.md) — W1 textbook deliverable (same series).
- `COURSE.md` §八 (W3) — the assignment statement.