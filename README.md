# From Token to Agent

> **From raw text to autonomous agents — building the modern LLM stack from scratch: tokenization, training, systems, scaling, post-training, RLVR, harnesses, and agents.**

> 🌐 **Languages**: [English](./README.md) · [简体中文](./README.zh-CN.md)

> 📌 **Course origin (read this first)**: This repo is the **course project for the 5-stage "LLM ladder" assignment** that **Prof. Jie Tang (唐杰)** formally assigned today — **2026-09-17, Tsinghua "Advanced Machine Learning"**. See [Course origin](#course-origin--2026-09-17-tsinghua) below for the full context, the PPT, and what the assignment grades on.

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

---

## Course origin — 2026-09-17, Tsinghua

This repository is **the working answer to a course assignment**, not an independent side project. Every choice in it — the 5-stage ladder, the 0.1B baseline, the Triton-attention requirement, the scaling-law hold-out, the controlled SFT/DPO/RLVR comparison, the verifiable-environment + harness + long-horizon agent + self-judge arc — comes from a single source:

> **Prof. Jie Tang (唐杰) — Tsinghua "Advanced Machine Learning" (高级机器学习), class of 2026 Fall, assignment handed out in lecture on 2026-09-17.**

### The assignment, in full

Prof. Tang's framing for the course project was: *"build the modern LLM stack from scratch, end to end, and let the previous stage's output be the next stage's input."* That sentence is what "A Ladder That Climbs the Course" means on the title slide. The 5 rungs of the ladder, in order:

1. **HW1 — Foundations.** Write a tokenizer + Transformer from scratch. End-to-end train a ~0.1B model on raw text. No HuggingFace Trainer as a shortcut.
2. **HW2 — Systems.** Hand-write a Triton attention kernel. Measure the gain yourself on single- and multi-GPU training and inference. The point isn't "learn Triton"; it's "freeze the model's outputs, then make it run faster."
3. **HW3 — Data + Scaling.** Start from a raw data dump. Build the full pipeline (parsing → language filter → quality filter → dedup → corpus). Train several smaller models. Fit a scaling law and **extrapolate to a held-out size you didn't train**. That's what turns the homework into a research artifact.
4. **HW4 — Post-Training.** From the **same** base model, run three controlled post-training routes — **SFT, DPO, RLVR** — and compare them on the **same** evaluation. The controlled comparison is the point: same base, same data budget, what does each route actually change?
5. **FINAL — Agents.** Build a **verifiable environment** (a coding env with unit tests is the cleanest starting point), a **harness** to run an agent inside it, train / optimize a **long-horizon agent**, and add a **self-judge** loop. The PPT encourages trajectory distillation, memory, and self-improvement as the research levers at this rung.

### Format and grading (from the lecture)

- **2–3 person teams.**
- **English paper, NeurIPS format.**
- **W16 in-class demo** (live run).
- **40% homework** (HW1–HW4); **60% final project** (FINAL).
- Every deliverable follows: **Problem → Motivation → Method → Results → Analysis → Reproduction**.

### Why this matters, and why it lives in *this* repo

The discipline is what differentiates this from "yet another LLM from-scratch tutorial":

- **The repo is the deliverable.** W16 the team demos the whole ladder in front of the class. There is no separate paper appendix repo.
- **Stages hand off code, not docs.** HW2 doesn't copy HW1's Transformer — it extends the same one with a Triton attention module under `src/kernels/`. HW3 doesn't fork a new tokenizer — it retrains the existing one on a new corpus. HW4 starts from HW3's checkpoint. FINAL inherits HW4's model.
- **Extensions, not forks.** New ideas don't open new repos. They open `extensions/EXT-NNN-*` folders that share `src/` with their parent stage and follow the same research template.
- **Boundaries are deliberately tight.** Five HW's. No HW5/HW6. The first three stages are for *understanding the moving parts*, not for winning on 0.1B benchmarks. The real research payoff starts at HW4 (RLVR, reward design, judge models) and ramps through FINAL (env design, harness, evaluation, self-improvement).

### Where the PPT lives

The lecture slide deck — including the *"A Ladder That Climbs the Course"* diagram and the *Problem → Motivation → Method → Results* footer — is the canonical source for this assignment. The author of this repo is Prof. Tang's student; the PPT is reproduced here for reference and as the seed for our team discussions.

> 🖼 **TODO (next commit):** embed the PPT slides as `docs/course-origin/tang-2026-09-17-aml-ladder.pdf` and a few key slide thumbnails (`docs/course-origin/slide-XX-ladder.png`) in this section.

---

See [README.zh-CN.md](./README.zh-CN.md) for the Chinese-language version.