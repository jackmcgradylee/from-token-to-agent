# HW1 — Foundations

> **Status:** **In Progress** (smoke test only). The full W1+W2+W3 deliverables (char + word tokenizer comparison, RMSNorm vs LayerNorm ablation, MHA vs GQA ablation, W3 model-side scaling pilot 20M/50M/100M) are not yet done. This HW will be tagged `v0.1-hw1-foundations` only after all pass criteria are met.

## Weeks covered

- **W1 — Three Paradigm Shifts:** `weeks/w01-paradigm-shifts/`
- **W2 — Architecture Revisited:** `weeks/w02-architecture-revisited/`
- **W3 — Training Dynamics & Scaling Laws:** `weeks/w03-training-dynamics-scaling-laws/`

## Implementation

This HW does not own any source code. The work is distributed across the codebase:

- **Tokenizer (W1):** `src/token_to_agent/tokenizer/`
  - `bpe.py` ✅ implemented
  - `char.py` ⏳ to do
  - `word.py` ⏳ to do
- **Transformer (W2):** `src/token_to_agent/model/`
  - `model.py`, `attention.py`, `rmsnorm.py`, `rope.py`, `swiglu.py`, `utils.py` ✅
  - `layernorm.py` ⏳ to do (for the RMSNorm vs LayerNorm ablation)
- **Training (W3):** `src/token_to_agent/training/`
  - `train.py`, `scheduler.py` ✅

## Experiments

- `experiments/w01/exp-001-char-word-bpe-comparison/` ⏳
- `experiments/w02/exp-001-layernorm-vs-rmsnorm/` ⏳
- `experiments/w02/exp-002-mha-vs-gqa/` ⏳
- `experiments/w03/exp-001-toy-baseline/` ✅ smoke only
- `experiments/w03/exp-002-scaling-pilot-20m/` ⏳
- `experiments/w03/exp-003-scaling-pilot-50m/` ⏳
- `experiments/w03/exp-004-scaling-pilot-100m/` ⏳

## Pass criteria (from COURSE.md §2 HW1)

A fresh environment can reproduce a short training run from one documented command.

## Reproduction

```bash
bash scripts/setup_env.sh
PYTHONPATH=src .venv/bin/python scripts/train.py --config configs/pretrain/toy-5m.yaml
PYTHONPATH=src .venv/bin/python scripts/generate.py \
    --ckpt artifacts/checkpoints/hw1-toy-5m/model.pt \
    --tokenizer artifacts/checkpoints/hw1-toy-5m/tokenizer.json \
    --prompts-file experiments/w03/exp-001-toy-baseline/prompts.txt \
    --out experiments/w03/exp-001-toy-baseline/samples.txt
PYTHONPATH=src .venv/bin/python tests/tokenizer/test_bpe.py
PYTHONPATH=src .venv/bin/python tests/model/test_transformer.py
```

## Full report

See `report.md` once HW1 reaches Pass Criteria.