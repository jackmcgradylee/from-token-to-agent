"""Plot the training loss curve from loss_curve.csv."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    train_df = df.dropna(subset=["train_loss"])
    val_df = df.dropna(subset=["val_loss"])

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(train_df) > 0:
        ax.plot(train_df["step"], train_df["train_loss"], label="train loss", marker=".", ms=4)
    if len(val_df) > 0:
        ax.plot(val_df["step"], val_df["val_loss"], label="val loss", marker="o", ms=6)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)