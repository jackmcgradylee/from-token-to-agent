# HW1 — Foundations: Build a LM from Scratch

> **One sentence:** Implement a BPE tokenizer and a Transformer LM from scratch, train it to ~0.1B parameters end-to-end on raw text, and produce samples — without `transformers.Trainer` or `nn.Transformer` shortcutting the learning.

---

## 1. Problem

Building a language model from scratch exposes every moving part: tokenization, embeddings, attention, training dynamics, sampling. Every later assignment (HW2 systems, HW3 data, HW4 post-training, FINAL agent) reuses code written here. We can't shortcut this stage.

**What I need to be able to do after HW1:**
- Read in raw text → produce a tokenizer that I trained.
- Read tokenizer output → run a forward pass through a Transformer I wrote myself.
- Run a training loop I wrote myself → produce a checkpoint.
- Load the checkpoint → generate text.

If any of those steps uses `transformers.AutoModelForCausalLM` or `datasets.load_dataset` to do the heavy lifting, HW1 is not done.

## 2. Motivation

- **HW2 Systems** replaces the attention path. That swap is only meaningful if I understand the path I'm replacing.
- **HW3 Data** retrains the tokenizer on a new corpus. That requires understanding what the tokenizer learned.
- **HW4 Post-Training** branches off the base checkpoint. The base must be under my control.
- **FINAL Agent** consumes the model. The model's failure modes are only interpretable if I trained it.

## 3. Method

```
raw text
   ↓
BPE tokenizer (src/tokenizer/)
   ↓
tokenized dataset (src/data/tinystories_dataset.py)
   ↓
Transformer LM (src/model/)
   ↓
training loop (src/training/)
   ↓
checkpoint (checkpoints/hw1/baseline-100m/)
   ↓
text generation (src/training/generate.py)
```

The baseline model is **`baseline-100m`** (~100M parameters, RoPE, RMSNorm, SwiGLU, GQA). All hyperparameters live in `configs/hw1/baseline-100m.yaml`. All later stages compare against this checkpoint.

## 4. Experimental Setup

### Hardware

- Reference: 1× NVIDIA A100 (80GB) or equivalent, bf16.
- Local fallback (this Jetson Nano, no GPU): toy-scale (~5M params) used to verify the pipeline end-to-end. Full-scale numbers below are reported when the repo is run on a GPU box.

### Data

- Primary: **TinyStories** (subset, English children's short stories, ~1M docs, vocab-friendly).
- Secondary (HW3 preview): Chinese Wikipedia subset.

### Model (baseline-100m)

| Field | Value |
|---|---|
| d_model | 768 |
| n_layers | 12 |
| n_heads | 12 |
| n_kv_heads | 4 (GQA: 3 query heads share 1 KV head) |
| d_ff | 2048 (SwiGLU → ~3072 internal) |
| vocab_size | 8192 |
| max_seq_len | 1024 |
| rope_theta | 10000 |
| init | Truncated normal (std 0.02) |
| params (approx) | ~95M |

### Training

| Field | Value |
|---|---|
| optimizer | AdamW (β=0.9, 0.95, ε=1e-8) |
| weight_decay | 0.1 |
| lr | 3e-4 |
| lr_schedule | cosine, 2000 warmup steps |
| batch_size | 8 × grad_accum 4 = effective 32 |
| seq_len | 1024 |
| tokens_target | ~1.5B |
| precision | bf16 (A100) / fp32 (Jetson fallback) |

## 5. Results

See `report.md` for the full Definition-of-Done table. Headline metrics must include:
- Tokenizer: vocab size, compression ratio, sample tokenization.
- Model: parameter count breakdown (by component).
- Training: train loss curve, val loss at end.
- Performance: tokens/s on reference hardware, peak memory.
- Generation: 5 fixed prompts, 200-token samples each.

## 6. Analysis

(Pre-fill outline; populated when the run completes.)

- Did loss converge to a reasonable range?
- Where did the model overfit / underfit?
- Did RoPE / GQA / SwiGLU / RMSNorm help? (HW1 → EXT-101/102/103/104 back-port.)
- What samples are good / bad? Diagnose by attention pattern.

## 7. Reproduction

```bash
# Set up env (Jetson / CPU)
bash scripts/setup_env.sh

# Train baseline-100m (GPU box; on Jetson see report.md for toy-scale alternative)
python scripts/train.py --config configs/hw1/baseline-100m.yaml

# Generate samples from a checkpoint
python scripts/generate.py \
    --ckpt checkpoints/hw1/baseline-100m/model.pt \
    --tokenizer checkpoints/hw1/baseline-100m/tokenizer.json \
    --prompts-file assignments/hw1-foundations/results/prompts.txt \
    --out assignments/hw1-foundations/results/samples.txt
```

## 8. Extensions

See `extensions/README.md` and the planned list:

- `EXT-101` RoPE (already part of baseline; extension = ablation without it)
- `EXT-102` RMSNorm (already part; ablation)
- `EXT-103` SwiGLU (already part; ablation)
- `EXT-104` GQA (already part; ablation vs MHA)
- `EXT-105` MuP init strategy
- `EXT-106` BBPE / Unigram tokenizer comparison

These all branch from `baseline-100m` — same data, same training budget, single change at a time.