# Course Schedule & Conventions

## 16-Week Schedule

| Week | Deliverable | Tag |
|---|---|---|
| W1–W3 | **HW1** Foundations — Build a LM from Scratch | `v0.1-hw1-foundations` |
| W4–W5 | **HW2** Systems — Triton Attention + Multi-GPU Benchmark | `v0.2-hw2-systems` |
| W6–W7 | **HW3** Data + Scaling — Corpus Pipeline + Scaling Law Hold-Out | `v0.3-hw3-scaling` |
| W8 | **Final Project Proposal** | `v0.4-proposal` |
| W9–W11 | **HW4** Post-Training — SFT vs DPO vs RLVR (controlled) | `v0.5-post-training` |
| W10 | **Final Mid Report** (overlaps with HW4) | `v0.6-midthesis` |
| W12–W15 | **FINAL** Agent — Environment + Harness + Long-Horizon Agent + Self-Judge | `v0.7-agent` |
| W16 | **Paper + Live Demo** | `v1.0-final` |

Weeks are loose. Don't gate on calendar — gate on **a working tag**.

---

## The two kinds of work: HW vs EXT

**HW (Homework)** — the course spine, must finish.

**EXT (Extension)** — incremental work on top of an HW. Has a strict numbering scheme:

```
EXT-<stage><index>

stage  = HW stage (1, 2, 3, 4, F)
index  = 3-digit zero-padded counter

Examples:
  EXT-101  HW1 extension #1   (e.g., RoPE)
  EXT-102  HW1 extension #2   (e.g., GQA)
  EXT-201  HW2 extension #1   (e.g., Triton attention)
  EXT-302  HW3 extension #2   (e.g., quality classifier)
  EXT-403  HW4 extension #3   (e.g., process reward)
  EXT-501  FINAL extension #1 (e.g., self-judge)
```

Every extension is **one folder** under `extensions/EXT-XXX-name/` with its own README using the same Problem → ... → Reproduction template. **Do not fork the repo for new ideas.** Open an EXT.

### Why this design

> "看到新的论文，不需要又新建一个 repo。
> 比如看到新的 Agent RL 方法：开 EXT-506，在现有环境和 Harness 上复现。
> 看到一个新的 Attention Kernel：开 EXT-207。"

This repo gradually becomes **your LLM research lab**, not your assignment graveyard.

---

## The research template (every README, every report)

Every `assignments/hwX/README.md`, `extensions/EXT-XXX/README.md`, and `paper/*/report.md` uses the same skeleton. It comes from 唐杰's PPT footer:

```
1. Problem       (what?)
2. Motivation    (why?)
3. Method        (how?)
4. Experimental Setup
5. Results
6. Analysis
7. Reproduction
8. Extensions
```

**坚持这个格式。** The byproduct is: you're training yourself to turn any engineering task into a research artifact.

---

## HW1 Definition of Done

| Surface | Must have |
|---|---|
| **Tokenizer** | vocab size, compression ratio, sample tokenization |
| **Model** | parameter count is computable from config |
| **Training** | train / val loss reported |
| **Performance** | tokens/s, peak memory |
| **Generation** | fixed-prompt samples saved to `results/` |
| **Reproduction** | one command re-trains the model |

**Lock in `checkpoints/hw1/baseline-100m/`** as the baseline. Do not modify it after W3. HW2/HW3/HW4 all branch from this checkpoint; changing it retroactively breaks every comparison.

---

## HW1 → HW2 → HW3 → HW4 → FINAL hand-off contract

```
HW1 outputs:
  src/tokenizer/  ──┐
  src/model/       ──┼── HW2 swaps in src/kernels/  (attention is the only replaced layer)
  src/training/    ──┘
  checkpoints/hw1/baseline-100m/  ──┐
                                    ├── HW3 retrains tokenizer on new corpus, refits scaling law
                                    ├── HW4 starts post-training from a chosen-scale checkpoint
                                    └── FINAL starts agent loop on top of HW4 model
```

If at any stage you find yourself copying code from a prior stage into a new folder, **stop**. Open an EXT instead, or refactor the prior stage.

---

## File & commit conventions

- `git tag -a vX.Y-<stage> -m "..."` at every milestone.
- One commit per logical unit ("add RoPE", not "W3 stuff").
- Commit messages: `hwN:` or `ext-XXX:` prefix, then one-line summary.
- Don't commit checkpoints to Git. They're under `checkpoints/` but `.gitignore`d, and we keep a SHA + metadata sidecar.

---

## What this course is NOT optimizing for

- SOTA on 0.1B models. The first 3 stages are for **understanding the moving parts**, not beating a benchmark.
- "Complete coverage" of LLM topics. We pick the 5 ladders; that's it.
- A general-purpose Agent framework on day one. FINAL = one verifiable env + one agent.

What it **is** optimizing for:

- A repo where every stage's output is the next stage's input.
- A research template you can apply to anything new you read.
- Extensions, not forks.