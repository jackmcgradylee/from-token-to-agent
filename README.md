# LLM From Scratch to Agent

> **One repo. One ladder. 16 weeks. From raw text to a self-improving agent.**

This repository is a single, continuously growing LLM research codebase. Every assignment builds on the previous one's output. The 5 formal assignments form the spine; everything else hangs off as numbered extensions.

```
Raw Data
   ↓
Tokenizer
   ↓
0.1B Base Model
   ↓
Systems Optimization
   ↓
Better Data / Scaling Law
   ↓
SFT / DPO / RLVR
   ↓
Verifiable Environment
   ↓
Harness
   ↓
Long-Horizon Agent
   ↓
Self-Judge
```

This is **not** five independent course projects. By W16, the entire repo is the artifact.

---

## The 5 Formal Assignments

| # | Title | Core task | Deliverable | Inputs the next stage uses |
|---|---|---|---|---|
| HW1 | Foundations — Build a LM from Scratch | Tokenizer + Transformer, end-to-end train a ~0.1B model | tokenizer, model code, training script, checkpoint, loss curves, generation samples, report | `src/tokenizer/`, `src/model/`, `src/training/`, `checkpoints/hw1/baseline-100m/` |
| HW2 | Systems — Make It Fast | Triton attention kernel + benchmark, multi-GPU train/infer | correctness, latency, throughput, memory, scaling efficiency | `src/kernels/`, replaces attention in `src/model/` |
| HW3 | Data + Scaling — Make Data Predictable | Build corpus pipeline, fit scaling law, hold-out predict | data pipeline, Data Card, scaling curve, prediction error | `src/data/`, tokenizer retrained, scaling extrapolation |
| HW4 | Post-Training — SFT vs DPO vs RLVR | 3 controlled routes from same base | 3 checkpoints, unified eval, training curves, cost/effect comparison | `src/post_training/`, base model from HW3 |
| FINAL | Agents — Environment + Harness + Long-Horizon Agent | Verifiable env + agent harness + self-judge | Proposal, Mid Report, Agent, Harness, Verifier, paper, live demo | `src/environments/`, `src/harness/`, `src/evaluation/` |

**Do not add HW5 / HW6.** New ideas go into `extensions/` numbered under the stage they belong to (`EXT-101`, `EXT-207`, `EXT-303`, `EXT-401`, `EXT-502`).

---

## Repository Layout

```
llm-from-scratch-to-agent/
├── README.md                  ← you are here
├── COURSE.md                  ← 16-week schedule, HW/EXT numbering, research template
│
├── assignments/               ← the "submit homework" view
│   ├── hw1-foundations/
│   ├── hw2-systems/
│   ├── hw3-data-scaling/
│   ├── hw4-post-training/
│   └── final-agent/
│
├── src/                       ← the actual code that keeps growing
│   ├── tokenizer/             ← BPE (HW1), re-trained (HW3)
│   ├── model/                 ← Transformer (HW1), attention swapped (HW2)
│   ├── training/              ← AdamW + LR schedule + checkpoint (HW1), reused everywhere
│   ├── kernels/               ← Triton attention etc. (HW2)
│   ├── data/                  ← corpus pipeline (HW3)
│   ├── post_training/         ← SFT/DPO/RLVR (HW4)
│   ├── environments/          ← Coding env, tests, verifier (FINAL)
│   ├── harness/               ← agent harness, tools, retry (FINAL)
│   └── evaluation/            ← unified eval (HW4, FINAL)
│
├── configs/                   ← YAML configs per HW
├── experiments/               ← per-stage experiment artifacts
├── checkpoints/               ← model weights, frozen at tagged milestones
├── extensions/                ← EXT-XXX incremental work, see extensions/README.md
├── reports/                   ← cross-cutting analysis
├── paper/                     ← proposal / midterm / final
├── scripts/                   ← CLI entry points
├── data/                      ← raw + processed data (gitignored)
└── tests/                     ← unit tests
```

### Why this split?

`assignments/` is the **submission view** — short, follows the Problem → Motivation → Method → Results → Analysis → Reproduction template (see `COURSE.md`).

`src/` is the **growing codebase**. HW2 does not copy HW1's Transformer; it extends the same one with a Triton attention module under `src/kernels/`. HW3 does not start a new tokenizer; it retrains the existing one on a new corpus. **Code accumulates**, never duplicates.

---

## Milestones & Git Tags

| Week | Milestone | Git tag |
|---|---|---|
| W1–W3 | HW1 Foundations | `v0.1-hw1-foundations` |
| W4–W5 | HW2 Systems | `v0.2-hw2-systems` |
| W6–W7 | HW3 Data + Scaling | `v0.3-hw3-scaling` |
| W8 | Final Project Proposal | `v0.4-proposal` |
| W9–W11 | HW4 Post-Training | `v0.5-post-training` |
| W10 | Final Mid Report | `v0.6-midthesis` |
| W12–W15 | Agent / Harness / Verifier | `v0.7-agent` |
| W16 | Paper + Live Demo | `v1.0-final` |

The GitHub timeline is therefore:

```
v0.1 → v0.2 → v0.3 → v0.4 → v0.5 → v0.6 → v0.7 → v1.0
```

Each tag is a snapshot of the **complete ladder working at that stage**. Not "only HW3 work merged".

---

## Headline Phrase

> **From raw text to a self-improving agent.**

Not marketing. Each line of that sentence is in the repo.

---

## Quick start

```bash
# Set up env (Jetson-friendly; see scripts/setup_env.sh)
bash scripts/setup_env.sh

# Reproduce HW1 baseline-100m (toy scale on Jetson, full scale on a GPU box)
python scripts/train.py --config configs/hw1/baseline-100m.yaml

# Generate samples
python scripts/generate.py --ckpt checkpoints/hw1/baseline-100m/ --prompt "Once upon a time"
```

See `assignments/hw1-foundations/README.md` for the full reproduction recipe.

---

## Boundaries (what this repo is NOT)

- **Not five isolated course projects.** Code accumulates.
- **Not chasing SOTA on 0.1B.** The first 3 stages are about *understanding*, not *winning*. Real research payoff starts at HW4 (RLVR) and the Final (environment / harness / eval).
- **Not "yet another Agent framework".** First version of FINAL is one verifiable environment + one harness + one agent. Generalization comes from extensions.