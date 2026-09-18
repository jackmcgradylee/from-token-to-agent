"""CLI entry: train a model from a YAML config (used by W3, W8–W11).

Reads:
  - config: configs/pretrain/<name>.yaml  (HW1)
  - config: configs/sft/<name>.yaml       (HW4 W8)
  - config: configs/dpo/<name>.yaml       (HW4 W9)
  - config: configs/rlvr/<name>.yaml      (HW4 W10–W11)

Usage:
    PYTHONPATH=src python scripts/train.py --config configs/pretrain/toy-5m.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
for p in (str(_repo_root), str(_repo_root / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml

from src.token_to_agent.from_scratch.tokenizer.bpe import BPETokenizer
from src.token_to_agent.from_scratch.model import TransformerLM, TransformerConfig, count_params
from src.token_to_agent.from_scratch.training.train import train, TrainConfig


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--tokenizer", default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)

    ckpt_dir = cfg.get("ckpt_dir", "artifacts/checkpoints/run")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Build tokenizer (or load existing)
    tok_path = args.tokenizer or str(Path(ckpt_dir) / "tokenizer.json")
    if Path(tok_path).exists():
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
        )
        tok.save(tok_path)
        print(f"[train] tokenizer saved to {tok_path}: {tok.stats_dict()}")

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