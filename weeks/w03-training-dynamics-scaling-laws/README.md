# W3 — Training Dynamics & Scaling Laws

## What this week is about

Why does a language model train successfully, and which quantities scale predictably as model size, data, and compute change?

## Status

In Progress. The full training stack is implemented and one toy run produced evidence. The W3 model-side scaling pilot (20M / 50M / 100M) and the formal scaling-law fit have not been done.

## Implemented so far

- `src/token_to_agent/training/train.py` — full training loop:
  - `TrainConfig` (lr, warmup, weight_decay, grad_clip, beta1, beta2, eps, batch_size, grad_accum, max_steps, log_every, eval_every, precision, seed)
  - `Trainer.train(model, train_ids, val_ids, cfg, device, log_fn)` — returns a summary dict
  - `TokenDataset` (slices in-memory token ids into (seq_len+1) windows)
  - AdamW with weight decay only on non-bias, non-norm params
  - Cosine LR schedule with linear warmup (`src/token_to_agent/training/scheduler.py`)
  - Gradient accumulation
  - CSV loss curve logging
  - Checkpoint save
- `src/token_to_agent/training/scheduler.py` — `cosine_lr(step, warmup, total, base_lr, min_lr_ratio)`
- `scripts/train.py` — YAML-config entry point
- `configs/pretrain/toy-5m.yaml` and `configs/pretrain/baseline-100m.yaml`

## Evidence

- `experiments/w03/exp-001-toy-baseline/` — smoke test:
  - Model: 4 layers × 256 d, 4 heads / 2 KV heads, GQA, SwiGLU, RoPE, RMSNorm
  - Tokens: ~386K (1 MB TinyStories-style synthetic text, 95/5 split)
  - Steps: 200, batch 4 × grad_accum 2 = effective 8
  - Train loss 6.43 → 2.27, val loss 2.26
  - Wall time ~292 s on Jetson CPU (no GPU)
  - Files: `loss_curve.png`, `prompts.txt`, `samples.txt`
  - Sample quality: local syntax + corpus vocabulary (`Lily`, `castle`, `dragon`) learned; long-range coherence weak (expected at 2.5M params)

## W3 model-side scaling pilot (pending)

- [ ] `experiments/w03/exp-002-scaling-pilot-20m/`
- [ ] `experiments/w03/exp-003-scaling-pilot-50m/`
- [ ] `experiments/w03/exp-004-scaling-pilot-100m/`
- [ ] N → loss, FLOPs → loss curves
- [ ] Document where the scaling curve fails

## Pass criteria (from COURSE.md §2 W3)

- [ ] A fresh environment can reproduce a short training run from one documented command. ✅ (smoke verified)
- [ ] Model-side scaling pilot completed with 20M / 50M / 100M.
- [ ] Scaling curve documented.

## Open questions

- The `configs/pretrain/toy-5m.yaml` config currently uses the toy CPU path; the full `configs/pretrain/baseline-100m.yaml` requires a GPU box. Should the W3 scaling pilot be run on a GPU box, or constrained to toy-scale on Jetson?
- The `precision: bf16` setting in baseline-100m requires CUDA. For the W3 pilot, should we add a `precision: fp32` variant?