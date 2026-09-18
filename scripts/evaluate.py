"""CLI entry: run a frozen evaluation suite against a model.

This is a thin entry point that:
  - loads a model from a checkpoint
  - loads the eval set (task suite JSONL by default)
  - runs each task and records metrics to a results JSONL
  - prints aggregate metrics

Eval tasks used by:
  - HW4 (W8–W11): same task distribution across SFT / DPO / RLVR / Agent RL
  - FINAL (W12–W16): frozen benchmark

Usage:
    PYTHONPATH=src python scripts/evaluate.py \
        --ckpt artifacts/checkpoints/hw1-toy-5m/model.pt \
        --tokenizer artifacts/checkpoints/hw1-toy-5m/tokenizer.json \
        --eval-set experiments/w03/exp-001-toy-baseline/prompts.txt \
        --out experiments/w03/exp-001-toy-baseline/eval.jsonl

This is a stub for W3 smoke. The full evaluator (success rate, pass@k, judge
calibration, safety eval, cost metrics) is built across W4 / W8 / W14 / W15.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
for p in (str(_repo_root), str(_repo_root / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from src.token_to_agent.from_scratch.tokenizer.bpe import BPETokenizer
from src.token_to_agent.from_scratch.model import TransformerConfig, TransformerLM


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--eval-set", required=True, help="Plain-text file of prompts, one per line.")
    p.add_argument("--out", required=True, help="Output JSONL of {prompt, completion, ...}")
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = TransformerConfig(**ckpt["cfg"])
    model = TransformerLM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for prompt in Path(args.eval_set).read_text(encoding="utf-8").splitlines():
            prompt = prompt.strip()
            if not prompt:
                continue
            ids = tok.encode(prompt)
            x = torch.tensor([ids], dtype=torch.long) if ids else torch.zeros(1, 1, dtype=torch.long)
            out = model.generate(x, max_new_tokens=args.max_tokens, use_cache=True)
            completion = tok.decode(out[0].tolist())
            f.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
            n += 1
    print(f"[eval] wrote {n} prompt/completion pairs to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())