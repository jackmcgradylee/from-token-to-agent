"""Shared harness for the W3 scaling-pilot sweep.

Each scaling point in `experiments/w03/exp-{002..006}-scaling-pilot-{NN}/`
calls `run_scaling_pilot(config_path, output_dir, prompts_path)` and gets
back a dict with the final val_loss, params count, tokens/s, wall time, and
the loss-curve CSV path. The thin `runner.py` in each exp folder just
fixes the paths and writes the 5-piece artefacts.

Run a single pilot from the repo root:

    PYTHONPATH=. .venv/bin/python \\
        experiments/w03/exp-002-scaling-pilot-1m/runner.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from src.token_to_agent.from_scratch.tokenizer.bpe import BPETokenizer
from src.token_to_agent.from_scratch.model import TransformerLM, TransformerConfig, count_params
from src.token_to_agent.from_scratch.training.train import train, TrainConfig


@dataclass
class PilotResult:
    target_params: int
    actual_params: int
    config_path: str
    final_train_loss: float
    final_val_loss: float
    tokens_per_sec_avg: float
    peak_mem_mb: float
    wall_seconds: float
    steps: int
    ckpt_dir: str
    loss_csv_path: str
    samples_path: str | None


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_loss_csv(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            rows.append({
                "step": int(row["step"]),
                "train_loss": float(row["train_loss"]) if row.get("train_loss") else None,
                "val_loss": float(row["val_loss"]) if row.get("val_loss") else None,
                "lr": float(row["lr"]) if row.get("lr") else None,
                "tokens_per_sec": float(row["tokens_per_sec"]) if row.get("tokens_per_sec") else None,
                "wall_s": float(row["wall_s"]) if row.get("wall_s") else None,
            })
    return rows


def _build_tokenizer(corpus_path: Path, cfg_tok: dict, save_path: Path) -> BPETokenizer:
    """Train a byte-level BPE on the corpus, or load an existing one.

    Two scaling points in the sweep share the same `vocab_size` and
    therefore share the same BPE merges. The harness trains it once
    and re-uses the saved file."""
    if save_path.exists():
        return BPETokenizer.load(str(save_path))
    text = corpus_path.read_text(encoding="utf-8")
    lines = [text]  # treat the whole file as one document for BPE
    tok = BPETokenizer.train(
        lines,
        target_vocab_size=cfg_tok["vocab_size"],
        min_pair_freq=cfg_tok.get("min_pair_freq", 2),
    )
    tok.save(str(save_path))
    return tok


def _generate_samples(model: TransformerLM, tok: BPETokenizer, prompts: list[str],
                      max_new_tokens: int, temperature: float, top_k: int | None) -> list[str]:
    """Greedy / top-k decode a list of prompts and return the continuations."""
    out_lines: list[str] = []
    for p in prompts:
        ids = tok.encode(p, add_bos=True, add_eos=False)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=max_new_tokens,
                               temperature=temperature, top_k=top_k, use_cache=True)
        out_lines.append(tok.decode(y[0].tolist()))
    return out_lines


import torch  # noqa: E402  (placed after the helper so the docstring reads cleanly)


def run_scaling_pilot(
    config_path: Path,
    output_dir: Path,
    prompts: list[str] | None = None,
    max_new_tokens: int = 30,
    temperature: float = 0.9,
    top_k: int = 50,
    device: str = "cpu",
) -> PilotResult:
    """Train one model from a W3 scaling config, write its 5-piece artefacts.

    Args:
        config_path: path to a YAML file matching `configs/hw1/scaling-*.yaml`.
        output_dir: where to write `loss_curve.csv`, `samples.txt`,
            `metadata.json`, `comparison.{md,json}`.
        prompts: optional list of seed prompts to decode after training. If
            `None`, a default 3-prompt set is used.
    """
    cfg = _load_yaml(config_path)
    ckpt_dir = cfg.get("ckpt_dir", f"checkpoints/{output_dir.name}")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Tokenizer: shared across the whole sweep (same vocab target).
    corpus_path = Path(cfg["data"]["path"])
    if not corpus_path.is_absolute():
        corpus_path = _REPO_ROOT / corpus_path
    tok_path = Path(ckpt_dir) / "tokenizer.json"
    tok = _build_tokenizer(corpus_path, cfg["tokenizer"], tok_path)

    # Model
    cfg["model"]["vocab_size"] = len(tok.id_to_bytes)
    model_cfg = TransformerConfig(**cfg["model"])
    model = TransformerLM(model_cfg)
    breakdown = count_params(model, by_component=True)
    actual_params = breakdown["_total"]
    target_params = cfg["model"].get("_target_params", actual_params)

    # Tokenize
    text = corpus_path.read_text(encoding="utf-8")
    all_ids = tok.encode(text, add_bos=True, add_eos=False)
    n_val = max(cfg["training"].get("val_size", 500), 100)
    train_ids = all_ids[: len(all_ids) - n_val]
    val_ids = all_ids[len(all_ids) - n_val:]

    # Train
    train_cfg = TrainConfig(**{k: v for k, v in cfg["training"].items() if k != "val_size"})
    train_cfg.ckpt_dir = ckpt_dir
    train_cfg.log_csv = cfg.get("log_csv", "loss_curve.csv")

    t0 = time.time()
    results = train(
        model,
        train_ids=train_ids,
        val_ids=val_ids,
        cfg=train_cfg,
        device=device,
        log_fn=lambda msg: print(f"[{output_dir.name}] {msg}"),
    )
    wall = time.time() - t0

    # Write loss curve to output_dir as well (in addition to ckpt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src_csv = Path(ckpt_dir) / train_cfg.log_csv
    dst_csv = output_dir / "loss_curve.csv"
    dst_csv.write_text(src_csv.read_text(encoding="utf-8"), encoding="utf-8")
    loss_rows = _read_loss_csv(dst_csv)

    # Generate samples (cheap, just for sanity).
    # NOTE: KV-cached generation in the bare model is slow per-token on CPU;
    #       we cap max_new_tokens and keep only 1 prompt to bound wall time.
    samples_path = output_dir / "samples.txt"
    if prompts is None:
        prompts = ["Lily "]
    samples = _generate_samples(model, tok, prompts, max_new_tokens, temperature, top_k)
    samples_path.write_text(
        "\n\n---\n\n".join(f"PROMPT: {p}\n{s}" for p, s in zip(prompts, samples)),
        encoding="utf-8",
    )

    # Optional loss-curve plot (gracefully skipped if matplotlib is missing).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps_tr = [r["step"] for r in loss_rows if r["train_loss"] is not None]
        loss_tr = [r["train_loss"] for r in loss_rows if r["train_loss"] is not None]
        steps_va = [r["step"] for r in loss_rows if r["val_loss"] is not None]
        loss_va = [r["val_loss"] for r in loss_rows if r["val_loss"] is not None]
        plt.figure(figsize=(7, 4))
        plt.plot(steps_tr, loss_tr, label="train", marker="o", markersize=3)
        plt.plot(steps_va, loss_va, label="val", marker="s", markersize=5)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(f"{output_dir.name}  N={actual_params:,d}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png", dpi=110)
        plt.close()
    except ImportError:
        pass

    return PilotResult(
        target_params=target_params,
        actual_params=actual_params,
        config_path=str(config_path),
        final_train_loss=results["final_train_loss"],
        final_val_loss=results["final_val_loss"],
        tokens_per_sec_avg=results["tokens_per_sec_avg"],
        peak_mem_mb=results["peak_mem_mb"],
        wall_seconds=wall,
        steps=results["steps"],
        ckpt_dir=ckpt_dir,
        loss_csv_path=str(dst_csv),
        samples_path=str(samples_path),
    )