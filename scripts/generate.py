"""CLI entry: generate samples from a trained checkpoint.

Usage:
    python scripts/generate.py \
        --ckpt checkpoints/hw1/toy-5m/model.pt \
        --tokenizer checkpoints/hw1/toy-5m/tokenizer.json \
        --prompts-file assignments/hw1-foundations/results/prompts.txt \
        --out assignments/hw1-foundations/results/samples.txt \
        --max-tokens 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable regardless of CWD.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import torch

from src.model import TransformerConfig, TransformerLM
from src.tokenizer.bpe import BPETokenizer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompts-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    print(f"[gen] loaded tokenizer: vocab={len(tok.id_to_bytes)}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = TransformerConfig(**ckpt["cfg"])
    model = TransformerLM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[gen] loaded model: {cfg.vocab_size} vocab, d_model={cfg.d_model}, "
          f"layers={cfg.n_layers}, heads={cfg.n_heads}, kv_heads={cfg.n_kv_heads}")

    prompts = Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
    prompts = [p for p in prompts if p.strip()]
    out_lines = []
    out_lines.append("# Generated samples")
    out_lines.append(
        f"# max_tokens={args.max_tokens}, temperature={args.temperature}, "
        f"top_k={args.top_k}, top_p={args.top_p}, use_cache={not args.no_cache}"
    )
    out_lines.append(f"# model: {cfg.n_layers}L x {cfg.d_model}d, "
                     f"{cfg.n_heads}H/{cfg.n_kv_heads}KV, vocab={cfg.vocab_size}")
    out_lines.append("")

    for i, prompt in enumerate(prompts, 1):
        ids = tok.encode(prompt)
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long)
        out = model.generate(
            x,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=not args.no_cache,
        )
        full_ids = out[0].tolist()
        # Decode everything (including the prompt).
        text = tok.decode(full_ids)
        out_lines.append(f"[p{i}] PROMPT: {prompt}")
        out_lines.append(f"[p{i}] FULL  : {text}")
        out_lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[gen] wrote {len(prompts)} samples to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())