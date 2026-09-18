# From Token to Agent · 从分词到智能体

> [🇬🇧 English](./README.en.md) · **🇨🇳 中文（默认）**
>
> From tokenization and pretraining to RLVR, long-horizon agents, and self-improvement —
> rebuilding the modern AI stack from scratch.

---

## Course origin

This repository is **the working answer to the 16-week assignment that Prof. Jie Tang (唐杰) formally assigned on 2026-09-17** in Tsinghua's *Advanced Machine Learning* lecture.

The course has only one real thesis, and uses 16 weeks to walk it all the way through—

> Supervised signal evolves from "human labels" to "self-judge". The technology stack is just the surface.

The "A Ladder That Climbs the Course" diagram at the bottom of Prof. Tang's PPT is the spine of this repo:

![Tang's 2026-09-17 AML lecture PPT collage — 6 slides in order: LLM use-case pyramid / Scaling↔Generalization timeline / opening framing "让机器像人一样思考" / 2026 research priorities / 16-week course map / historical context — three paradigm shifts, 1980s → 2026](./docs/course-origin/tang-2026-09-17-aml-ppt-collage.jpg)

The six slides, left-to-right: LLM use-case pyramid · Scaling ↔ Generalization timeline · Opening framing "让机器像人一样思考" (Machines that think like humans) · 2026 research priorities · **the 16-week course map** (which this repo's progress table mirrors) · historical context (three paradigm shifts, 1980s → 2026).

The full 16-week syllabus — weekly deliverables, pass criteria, extension conventions — is frozen in [`COURSE.md`](./COURSE.md).

---

## Learning Signal Ladder

```
labels              ←  W1   Supervised labels: discretize the world
word structure      ←  W1   Word structure: inductive biases of BPE / Word / Char
next-token          ←  W1-W7  Predict the next token: Transformer + Scaling Law
preferences         ←  W8-W10 SFT / DPO: align to human preference
verifiers           ←  W10-W11 RLVR / ORM / PRM: verifiable rewards
environment         ←  W12-W13 Coding Agent / Memory
self-judge          ←  W14-W16 Self-judge / trajectory distillation / long-horizon autonomy
```

Each rung = signal gets cheaper to produce, humans exit one more loop.

---

## Progress (as of W5)

| Stage | Status | Milestone |
|---|---|---|
| **HW1 Foundations** — implement Tokenizer + Transformer from scratch, end-to-end train ~0.1B | 🟡 in progress | toy-5m smoke done, scaling to baseline-100m |
| **HW2 Systems** — hand-write Triton Attention Kernel, benchmark train / inference / multi-GPU | 🟡 in progress | Triton attention kernel + benchmark |
| **HW3 Data + Scaling** — build Corpus from Raw Dump, fit Scaling Law and verify extrapolation | ⚪ not started | W6 |
| **HW4 Post-Training** — compare SFT / DPO / RLVR on same Base Model, extend to Agent RL | ⚪ not started | W8 |
| **Final Project** — Verifiable Environment + Harness + Long-Horizon Agent + Self-Judge | ⚪ not started | proposal starts W8 |

**Currently at**: W4 — compute kernels & parallelism.

See [ROADMAP.md](./ROADMAP.md).

---

## 5 assignments ↔ 16 weeks

The course walks **5 assignments** through **16 weeks** with a 1-to-1 mapping:

| Assignment | Weeks | Topic | Core deliverable |
|---|---|---|---|
| **HW1 Foundations** | W1 – W3 | Three paradigm shifts / Decoder-only Transformer / Training dynamics & scaling | Tokenizer + Transformer + Training Loop + ~0.1B Pretraining + Architecture Ablation |
| **HW2 Systems** | W4 – W5 | Compute / Kernels / Parallelism / Inference economics | Triton Kernel + Multi-GPU + Serving Benchmark + Inference Cost |
| **HW3 Data + Scaling** | W6 – W7 | Pretraining data / Synthetic data & governance / Scaling Law extrapolation | Raw Pipeline + Data Card + Synthetic + Scaling Law + Held-out Prediction |
| **HW4 Post-Training** | W8 – W11 | SFT & distillation / Preference / RLVR / Agent RL long-horizon | SFT + DPO + RLVR + Agentic Extension (same Base / same Eval / same Cost) |
| **Final Project** | W8 – W16 | Verifiable Environment + Harness + Long-Horizon Agent + Self-Judge | Proposal (W8) → Mid Report (W10) → Final Paper + Live Demo (W16) |

Milestone submission view: [`assignments/`](./assignments/). Weekly learning notes: [`weeks/`](./weeks/).

---

## What this repo is

**Not 4 independent assignments.** It's one continuously upgrading system
that walks from `raw text` to a `self-improving agent`:

```
raw data
  ↓
tokenizer (W1)
  ↓
0.1B base model (W2-W3)
  ↓
systems optimization (W4-W5)
  ↓
better data + scaling (W6-W7)
  ↓
SFT / DPO / RLVR (W8-W11)
  ↓
verifiable environment (W12)
  ↓
harness + long-horizon agent (W12-W13)
  ↓
self-judge + trajectory distillation (W14-W16)
```

Each week's code inherits the previous week's artifacts, so after 16 weeks
**the repo itself is the work**.

---

## Repo layout

```
from-token-to-agent/
├── weeks/              ★ Primary axis: 16-week learning rhythm (W1-W5 created)
├── assignments/        ★ Secondary axis: milestone submission view
├── src/token_to_agent/ ★ Sole canonical code region
│   ├── from_scratch/   W1-W4 + W10 core hand-written implementations
│   │   ├── tokenizer/    BPE (W1)
│   │   ├── model/        decoder-only Transformer (W2)
│   │   ├── training/     AdamW + cosine LR + checkpoint (W3)
│   │   ├── kernels/      Triton + reference attention (W4)
│   │   ├── dpo/          DPO core loss (W9 teaching version)
│   │   └── rlvr/         RLVR teaching loop (W10)
│   ├── systems/        W4-W5 framework wrappers
│   │   ├── distributed/    DDP / FSDP / Megatron glue
│   │   ├── serving/        vLLM / SGLang client + Load Generator
│   │   └── profiling/      kernel breakdown + arithmetic intensity
│   ├── data/           W6-W7 Raw pipeline + Synthetic Data Factory
│   ├── post_training/  W8-W11 scale-train glue (TRL / slime / verl)
│   │   ├── sft/            SFT runner
│   │   ├── preference/     DPO scale runner
│   │   └── rl/             RLVR + Agent RL runner
│   ├── agent/          W12-W16 hand-written agent stack
│   │   ├── harness/        minimum Harness (Parser / Context / Retry / Budget / Sandbox)
│   │   ├── environments/   Verifiable Coding Agent envs
│   │   ├── tools/          Tool schemas + parsers
│   │   ├── memory/         Working / Episodic / Procedural
│   │   └── self_eval/      Self-Judge + Calibration
│   └── evaluation/     HW4 + FINAL unified eval matrix
├── tests/              per-module
├── configs/            training / systems / post-training / agent configs
├── experiments/        5-piece: config + metadata + metrics + README per experiment
├── extensions/         incremental experiments off the main line (EXT-W<N>-<idx>)
├── docs/               course origin / architecture / paper reading notes
├── artifacts/          local artifacts (gitignored)
├── scripts/            5 CLIs: train / evaluate / generate / benchmark / serve
├── COURSE.md           frozen 16-week syllabus
└── ROADMAP.md          progress snapshot
```

---

## Quick start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run tests
bash tests/test.sh

# Train toy 0.1B
bash scripts/setup_env.sh
python scripts/train.py --config configs/pretrain/toy-5m.yaml

# Generate
python scripts/generate.py --checkpoint artifacts/checkpoints/toy-5m --prompt "你好"

# Benchmark (W2)
python scripts/benchmark.py attention --backend triton --seq-len 128 1024
```

> ⚠️ Code is **complete and runnable**, but 0.1B-scale experiments need a GPU host.
> Jetson Nano is dev / teaching only; toy-5m runs on CPU in ~5 min.

---

## 16-week rhythm

| Week | Deliverable | Status |
|---|---|---|
| W1-W3 | HW1 Foundations | 🟡 in progress |
| W4-W5 | HW2 Systems | 🟡 in progress |
| W6-W7 | HW3 Data + Scaling | ⚪ not started |
| W8 | Final Project Proposal | ⚪ not started |
| W9-W11 | HW4 Post-Training | ⚪ not started |
| W10 | Final Mid-Report | ⚪ not started |
| W12-W15 | Agent / Harness / Verifier | ⚪ not started |
| W16 | Paper + Live Demo | ⚪ not started |

Git tag per stage: `v0.1-hw1-foundations` / `v0.2-hw2-systems` / … / `v1.0-from-token-to-agent`.

---

## License

Apache 2.0 — see [LICENSE](./LICENSE).

---

## Acknowledgements

- **Course source**: Prof. Jie Tang (唐杰) — *Advanced Machine Learning*, 2026-09-17 — [PPT collage](./docs/course-origin/tang-2026-09-17-aml-ppt-collage.jpg)
- **Methodology**: HKUDS/CLI-Anything (agent-native CLI design)
- **Paper references**: THUDM GLM series (GLM-4.5 / GLM-5 / slime / DeepDive / ReST-MCTS etc.)