"""W3 — scaling-law fit + hold-out prediction (exp-007).

Reads the metadata.json of all 5 W3 scaling-pilot experiments (1M / 2.5M /
5M / 10M / 20M), fits the Chinchilla-style power law

    L(N) = L_inf + a * N ** (-alpha)

on the 4 non-hold-out points, then **predicts** the held-out point and
measures the prediction error. Hold-out choice (per user spec) is the
largest model (20M) — the test of "do my fits extrapolate to a model I
never trained in the fit set?".

Reproduce:
    PYTHONPATH=. .venv/bin/python experiments/w03/exp-007-scaling-law-fit/runner.py

Output:
    - results/fit.json       fitted (a, alpha, L_inf), per-point residual
    - results/fit.md         markdown report
    - results/scaling_curve.png   loss vs N (log-x, log-y), with fit + hold-out marker
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent

# Hold-out point: 20M-class. Fit on the 4 smaller points.
HOLDOUT_DIR = _REPO_ROOT / "experiments/w03/exp-006-scaling-pilot-20m"
FIT_DIRS = [
    _REPO_ROOT / "experiments/w03/exp-002-scaling-pilot-1m",
    _REPO_ROOT / "experiments/w03/exp-003-scaling-pilot-2_5m",
    _REPO_ROOT / "experiments/w03/exp-004-scaling-pilot-5m",
    _REPO_ROOT / "experiments/w03/exp-005-scaling-pilot-10m",
]


@dataclass
class PilotPoint:
    label: str          # "1m", "2_5m", ...
    n_params: int       # actual parameter count
    val_loss: float     # final val loss from the pilot's metadata.json


def load_pilot(d: Path) -> PilotPoint:
    md = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    return PilotPoint(
        label=d.name.rsplit("-", 1)[-1],
        n_params=md["actual_params"],
        val_loss=md["final_val_loss"],
    )


def fit_power_law(points: list[PilotPoint]) -> tuple[float, float, float]:
    """Fit log(L - L_inf) = log(a) - alpha * log(N) by grid search over L_inf.

    Closed-form least squares in log-space would be cleaner, but with only
    4 points a 1-D grid search over `L_inf` keeps the model identifiable
    and avoids the `log(L - L_inf)` being undefined when `L_inf > L(N)`.

    Returns: (a, alpha, L_inf) — `L(N) = L_inf + a * N^(-alpha)`.
    """
    best: tuple[float, float, float, float] | None = None
    log_n = [math.log(p.n_params) for p in points]
    loss = [p.val_loss for p in points]
    loss_min = min(loss)
    # Grid L_inf from 0.5*loss_min up to 0.99*loss_min in 50 steps.
    for li_frac in [0.5 + 0.5 * i / 50 for i in range(50)]:
        L_inf = loss_min * li_frac
        shifted = [l - L_inf for l in loss]
        if any(s <= 0 for s in shifted):
            continue
        log_s = [math.log(s) for s in shifted]
        # Linear fit: y = c + m * x ; c=log(a), m=-alpha
        n = len(points)
        sx = sum(log_n)
        sy = sum(log_s)
        sxx = sum(x * x for x in log_n)
        sxy = sum(x * y for x, y in zip(log_n, log_s))
        denom = n * sxx - sx * sx
        if denom == 0:
            continue
        m = (n * sxy - sx * sy) / denom
        c = (sy - m * sx) / n
        alpha = -m
        log_a = c
        a = math.exp(log_a)
        # Sum of squared residuals in original space
        residuals = [(L_i - (L_inf + a * math.exp(-alpha * math.log(p.n_params)))) ** 2
                    for L_i, p in zip(loss, points)]
        ssr = sum(residuals)
        if best is None or ssr < best[3]:
            best = (a, alpha, L_inf, ssr)
    if best is None:
        raise RuntimeError("grid search failed")
    return best[0], best[1], best[2]


def predict(a: float, alpha: float, L_inf: float, n: int) -> float:
    return L_inf + a * n ** (-alpha)


def render_markdown(
    pilots: list[PilotPoint],
    a: float, alpha: float, L_inf: float,
    holdout: PilotPoint | None,
    predicted: float,
    actual: float,
    rel_err: float,
) -> str:
    lines = []
    lines.append("# W3 — Scaling-law fit + hold-out prediction\n")
    lines.append("Fit the Chinchilla-style power law on the 4 non-hold-out points, "
                 "then predict the held-out point (the largest model, 20M-class).\n")
    lines.append("## Pilot summary\n")
    lines.append("| Label | N (params) | Val loss |")
    lines.append("| --- | ---: | ---: |")
    for p in pilots:
        marker = "  *(hold-out)*" if (holdout is not None and p.label == holdout.label) else ""
        lines.append(f"| `{p.label}` | {p.n_params:,d} | {p.val_loss:.4f}{marker} |")
    lines.append("")
    lines.append("## Fit\n")
    lines.append("Model: `L(N) = L_inf + a · N^(-α)`. Fitted by 1-D grid search "
                 "over `L_inf` with closed-form least squares for `a` and `α`.\n")
    lines.append(f"- **L_inf** = `{L_inf:.4f}`")
    lines.append(f"- **a**     = `{a:.4f}`")
    lines.append(f"- **α**     = `{alpha:.4f}` (exponent on N)")
    lines.append("")
    if holdout is None:
        lines.append("## Hold-out prediction\n")
        lines.append("**Hold-out point (20M) not yet trained.** "
                     "Re-run `experiments/w03/exp-007-scaling-law-fit/runner.py` "
                     "after `exp-006-scaling-pilot-20m` finishes to get the "
                     "extrapolation error.\n")
        lines.append("## What this means\n")
        lines.append("Fit is computed on the 4 small points; the 20M hold-out "
                     "comparison will be filled in when that pilot completes.\n")
        lines.append("## Caveats\n")
        lines.append("- **Iso-data, not iso-compute**: each pilot trained on the same ")
        lines.append("  ~386K-token corpus. This isolates the model-size effect, but ")
        lines.append("  real scaling laws (Hoffmann 2022) optimize compute jointly.")
        lines.append("- **100-step budget**: convergence is far below what production ")
        lines.append("  sweeps use. The `L(N)` curve here describes *partially-trained* ")
        lines.append("  loss, not converged loss.")
        lines.append("- **Tiny corpus (1 MB)**: extrapolation outside this regime is ")
        lines.append("  not informative.")
        lines.append("")
        lines.append("## Files\n")
        lines.append("- [`results/fit.json`](./results/fit.json)")
        lines.append("- [`results/scaling_curve.png`](./results/scaling_curve.png)")
        lines.append("- [`runner.py`](./runner.py)")
        return "\n".join(lines)

    lines.append("## Hold-out prediction\n")
    lines.append("| Held-out | N | Predicted loss | Absolute error | Relative error |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    lines.append(f"| `{holdout.label}` | {holdout.n_params:,d} | {predicted:.4f} | "
                 f"{abs(actual - predicted):.4f} | {rel_err:.2%} |")
    lines.append("")
    lines.append("## What this means\n")
    if rel_err < 0.10:
        verdict = "The fit extrapolates well. The loss-vs-N curve on this corpus "\
                  "behaves like a clean power law over the swept range."
    elif rel_err < 0.25:
        verdict = "The fit extrapolates within 25%. Decent, but the curve may be "\
                  "bending (deep models under-trained, or training budget "\
                  "limited)."
    else:
        verdict = "The fit extrapolates poorly. Either the 20M point is not on the "\
                  "same scaling regime as the smaller points, or our training "\
                  "budget (100 steps, ~386K tokens) is too small to converge."
    lines.append(verdict + "\n")
    lines.append("## Caveats\n")
    lines.append("- **Iso-data, not iso-compute**: each pilot trained on the same ")
    lines.append("  ~386K-token corpus. This isolates the model-size effect, but ")
    lines.append("  real scaling laws (Hoffmann 2022) optimize compute jointly.")
    lines.append("- **100-step budget**: convergence is far below what production ")
    lines.append("  sweeps use. The `L(N)` curve here describes *partially-trained* ")
    lines.append("  loss, not converged loss.")
    lines.append("- **Tiny corpus (1 MB)**: extrapolation outside this regime is ")
    lines.append("  not informative.")
    lines.append("")
    lines.append("## Files\n")
    lines.append("- [`results/fit.json`](./results/fit.json)")
    lines.append("- [`results/scaling_curve.png`](./results/scaling_curve.png)")
    lines.append("- [`runner.py`](./runner.py)")
    return "\n".join(lines)


def main() -> int:
    # Load fit-set pilots (4 small ones).
    fit_points: list[PilotPoint] = [load_pilot(d) for d in FIT_DIRS]
    # Hold-out point — load lazily; if its metadata isn't there yet, we
    # fit & report without a hold-out comparison.
    holdout: PilotPoint | None = None
    holdout_meta = HOLDOUT_DIR / "metadata.json"
    if holdout_meta.is_file():
        holdout = load_pilot(HOLDOUT_DIR)
    pilots: list[PilotPoint] = list(fit_points) + ([holdout] if holdout else [])

    a, alpha, L_inf = fit_power_law(fit_points)
    if holdout is not None:
        predicted = predict(a, alpha, L_inf, holdout.n_params)
        actual = holdout.val_loss
        rel_err = abs(actual - predicted) / actual
    else:
        predicted = float("nan")
        actual = float("nan")
        rel_err = float("nan")

    (OUTPUT_DIR / "results").mkdir(exist_ok=True)

    fit_payload = {
        "model": "L(N) = L_inf + a * N^(-alpha)",
        "fit_points": [
            {"label": p.label, "n_params": p.n_params, "val_loss": p.val_loss}
            for p in fit_points
        ],
        "holdout_point": (
            {"label": holdout.label, "n_params": holdout.n_params,
             "actual_val_loss": holdout.val_loss}
            if holdout is not None else None
        ),
        "fit_params": {"a": a, "alpha": alpha, "L_inf": L_inf},
        "prediction": (
            {"predicted_val_loss": predicted,
             "abs_error": abs(actual - predicted),
             "rel_error": rel_err}
            if holdout is not None else None
        ),
    }
    (OUTPUT_DIR / "results" / "fit.json").write_text(
        json.dumps(fit_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "results" / "fit.md").write_text(
        render_markdown(pilots, a, alpha, L_inf, holdout, predicted, actual, rel_err),
        encoding="utf-8",
    )

    # Plot — gracefully skipped if matplotlib is missing.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [p.n_params for p in pilots]
        ys = [p.val_loss for p in pilots]
        # Dense grid for the fit line
        x_grid = [10 ** (math.log10(min(xs)) + (math.log10(max(xs)) - math.log10(min(xs))) * i / 200)
                  for i in range(201)]
        y_grid = [predict(a, alpha, L_inf, x) for x in x_grid]
        plt.figure(figsize=(7, 5))
        plt.loglog(x_grid, y_grid, label=f"fit  L(N)=L_inf+a·N^(-α), α={alpha:.3f}", color="C0")
        for p in pilots:
            color = "C3" if p.label == "20m" else "C2"
            marker = "*" if p.label == "20m" else "o"
            size = 200 if p.label == "20m" else 80
            plt.scatter([p.n_params], [p.val_loss], color=color, marker=marker,
                        s=size, zorder=3,
                        label=f"{p.label}  (val={p.val_loss:.3f})" if p.label != "20m"
                              else f"hold-out {p.label}  actual={p.val_loss:.3f}")
            if p.label == "20m" and holdout is not None and not math.isnan(predicted):
                # also plot the prediction as an open circle
                plt.scatter([p.n_params], [predicted], facecolors="none",
                            edgecolors="C3", s=300, zorder=2,
                            label=f"hold-out {p.label}  predicted={predicted:.3f}")
        plt.xlabel("N (params)")
        plt.ylabel("val_loss")
        plt.title("W3 scaling-law fit + hold-out prediction (20M held out)")
        plt.legend(fontsize=8, loc="best")
        plt.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "results" / "scaling_curve.png", dpi=110)
        plt.close()
    except ImportError:
        pass

    print(f"[exp-007] fit: a={a:.4f}  alpha={alpha:.4f}  L_inf={L_inf:.4f}")
    if holdout is not None:
        print(f"[exp-007] hold-out {holdout.label} (N={holdout.n_params:,d}): "
              f"actual={actual:.4f}  predicted={predicted:.4f}  rel_err={rel_err:.2%}")
    else:
        print(f"[exp-007] hold-out 20M not trained yet; fit reported without "
              f"hold-out comparison (re-run after 20M completes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())