# From Token to Agent

> Building the modern AI stack from scratch:
> from tokenization and pretraining to RLVR,
> long-horizon agents and self-improvement.

---

## Learning Signal Ladder

```
labels
→ word structure
→ next-token
→ preferences
→ verifiers
→ environment
→ self-judge
```

This is the ladder implied by Prof. Jie Tang's 16-week course at Tsinghua "Advanced Machine Learning" (高级机器学习, 2026-09-17). Each rung is "the signal gets cheaper to produce — humans exit one more loop."

The PPT collage from that lecture is reproduced below.

![Tang's 2026-09-17 lecture slides — 6-slide collage, including the actual 16-week course map and the 7-rung ladder footer](./docs/course-origin/tang-2026-09-17-aml-ppt-collage.jpg)

The collage shows six slides in order: the LLM use-case pyramid, the Scaling ↔ Generalization timeline, the opening framing ("让机器像人一样思考"), the 2026 research priorities, the **16-week course map**, and the historical context (three paradigm shifts, 1980s → 2026).

## Progress

```
W01  Three Paradigm Shifts                   Done (smoke)
W02  Architecture Revisited                  In Progress
W03  Training Dynamics & Scaling Laws        In Progress
W04  Compute, Kernels, Parallelism           In Progress
W05  Economics of Inference                  Not Started
W06  Pretraining Data                        Not Started
W07  Synthetic Data & Governance             Not Started
W08  SFT & Distillation                      Not Started
W09  Preference Learning                     Not Started
W10  RLVR & Reasoning                        Not Started
W11  Agent RL: Long-Horizon                  Not Started
W12  Agent Foundations                       Not Started
W13  Memory & Continual Learning             Not Started
W14  Self-Evaluation & Evolution             Not Started
W15  Evaluation & Safety                     Not Started
W16  What Comes After LLMs?                  Not Started
```

```
HW1 — Foundations          In Progress
HW2 — Systems              In Progress
HW3 — Data + Scaling       Not Started
HW4 — Post-Training        Not Started
Final Project              Not Started
```

See [`ROADMAP.md`](./ROADMAP.md) for the detailed current state and next concrete steps.

## Course

The full 16-week plan, weekly deliverables, pass criteria, and extension conventions live in [`COURSE.md`](./COURSE.md). That file is the source of truth for the syllabus and is treated as frozen.

## Repository

```
weeks/        learning progression — W1 → W16, one folder per week
assignments/  milestone submissions — HW1, HW2, HW3, HW4, Final
src/          the single codebase that keeps growing
              organized as src/token_to_agent/<module>/
tests/        unit and correctness tests, mirrored to src/
configs/      reproducible experiment configurations
experiments/  every real run's evidence (config, metrics, metadata, README)
extensions/   research beyond the course baseline
              numbered EXT-W<N>-<idx>, one folder per extension
docs/         architecture diagrams, paper reading notes, course-origin artifacts
artifacts/    local-only outputs (checkpoints, datasets, logs, traces, profiles)
scripts/      CLI entry points (train, evaluate, benchmark, generate, serve)
```

The discipline: **code accumulates, never duplicates.** A week writes into `src/token_to_agent/<module>/`. An HW `report.md` only references code and experiment evidence — it does not copy either.

## Quick start

```bash
bash scripts/setup_env.sh

# Train the toy 0.1B scratch baseline (smoke test, ~5 min on Jetson CPU)
PYTHONPATH=. .venv/bin/python scripts/train.py --config configs/pretrain/toy-5m.yaml

# Generate samples
PYTHONPATH=. .venv/bin/python scripts/generate.py \
    --ckpt artifacts/checkpoints/hw1-toy-5m/model.pt \
    --tokenizer artifacts/checkpoints/hw1-toy-5m/tokenizer.json \
    --prompts-file experiments/w03/exp-001-toy-baseline/prompts.txt \
    --out experiments/w03/exp-001-toy-baseline/samples.txt

# Run all tests
PYTHONPATH=. .venv/bin/python tests/tokenizer/test_bpe.py
PYTHONPATH=. .venv/bin/python tests/model/test_transformer.py
PYTHONPATH=. .venv/bin/python tests/kernels/test_attention.py
```

## Boundaries

- **Not 16 isolated weekly projects.** Code accumulates; weeks are learning units, not submission units.
- **Not chasing SOTA on 0.1B.** The first 7 weeks are for *understanding the moving parts*, not for winning on small-model benchmarks. Real research payoff starts at W8 (SFT/DPO/RLVR) and ramps through W16 (env / harness / eval / self-improvement).
- **Not "yet another Agent framework".** First version of the Final block is one verifiable environment + one harness + one agent. Generalization comes from extensions.
- **No download of pretrained weights or large datasets.** This repo is implement-first, framework-second — the mechanisms are written here, not fetched.

## License

Apache 2.0 — see [`LICENSE`](./LICENSE).

## Acknowledgements

This repository is **the working answer to the 16-week course assignment** that **Prof. Jie Tang (唐杰)** formally assigned on **2026-09-17** in Tsinghua's "Advanced Machine Learning" lecture. The 7-rung ladder, the Problem → Motivation → Method → Results research template, the "code accumulates, never duplicates" discipline, and the "no HW5/HW6 — extensions only" rule are all directly from that assignment.

The reading spine in `COURSE.md §7` references work by Tang's group (THUDM) and Z.ai — GLM-4.5, GLM-5, slime, ReST-MCTS*, TDRM, AgentTuning, AgentBench, ScaleCUA, INFTY — as orientation, not as things to reproduce wholesale.