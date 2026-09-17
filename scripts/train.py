"""CLI entry: train the model from a YAML config.

Usage:
    python scripts/train.py --config configs/hw1/toy-5m.yaml

The config file holds model + training hyperparameters + tokenizer path.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# Make `src/` importable regardless of CWD.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.model import TransformerLM, TransformerConfig, count_params  # noqa: E402
from src.tokenizer.bpe import BPETokenizer  # noqa: E402
from src.training.train import train, TrainConfig  # noqa: E402


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument(
        "--tokenizer",
        default=None,
        help="Override tokenizer path; default = ckpt_dir/tokenizer.json",
    )
    args = p.parse_args()

    cfg = load_yaml(args.config)
    cfg_path = Path(args.config).resolve()

    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/run")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build tokenizer
    tok_path = args.tokenizer or os.path.join(ckpt_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        print(f"[train] loading tokenizer from {tok_path}")
        tok = BPETokenizer.load(tok_path)
    else:
        corpus_path = cfg["data"]["path"]
        print(f"[train] training tokenizer on {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        tok = BPETokenizer.train(
            lines,
            target_vocab_size=cfg["tokenizer"]["vocab_size"],
            min_pair_freq=cfg["tokenizer"].get("min_pair_freq", 2),
            verbose=True,
        )
        tok.save(tok_path)
        print(f"[train] tokenizer saved to {tok_path}: {tok.stats_dict()}")

    # Update model vocab_size to match tokenizer.
    cfg["model"]["vocab_size"] = len(tok.id_to_bytes)
    model_cfg = TransformerConfig(**cfg["model"])
    model = TransformerLM(model_cfg)
    breakdown = count_params(model, by_component=True)
    print(f"[train] model params: {breakdown['_total']:,d}")
    for k, v in breakdown.items():
        if k == "_total":
            continue
        print(f"        {k:14s} {v:>10,d}")

    # Tokenize corpus
    print(f"[train] tokenizing {cfg['data']['path']}")
    with open(cfg["data"]["path"], "r", encoding="utf-8") as f:
        all_text = f.read()
    all_ids = tok.encode(all_text, add_bos=True, add_eos=False)
    n = len(all_ids)
    n_val = max(cfg["training"].get("val_size", 5000), 1000)
    train_ids = all_ids[: n - n_val]
    val_ids = all_ids[n - n_val :]
    print(f"[train] tokens: {n:,d}  train={len(train_ids):,d}  val={len(val_ids):,d}")

    train_cfg = TrainConfig(**{k: v for k, v in cfg["training"].items() if k != "val_size"})
    train_cfg.ckpt_dir = ckpt_dir
    train_cfg.log_csv = cfg.get("log_csv", "loss_curve.csv")

    results = train(
        model,
        train_ids=train_ids,
        val_ids=val_ids,
        cfg=train_cfg,
        device=args.device,
        log_fn=lambda msg: print(f"[train] {msg}"),
    )
    print("[train] done:", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())