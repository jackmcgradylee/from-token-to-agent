# Roadmap — Where this repo is right now

> Last updated: 2026-09-17

This file tracks real progress through the 16-week course. It is intentionally short — read [`COURSE.md`](./COURSE.md) for the full plan, the weekly deliverables, the assessment rules, and the research reading spine.

## Progress

### Weekly modules

```
W01  Three Paradigm Shifts                   Done (smoke)  — see weeks/w01-paradigm-shifts/
W02  Architecture Revisited                  In Progress   — see weeks/w02-architecture-revisited/
W03  Training Dynamics & Scaling Laws        In Progress   — see weeks/w03-training-dynamics-scaling-laws/
W04  Compute, Kernels, Parallelism           In Progress   — see weeks/w04-compute-kernels-parallelism/
W05  Economics of Inference                  Not Started   — weeks/w05-economics-of-inference/
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

### Homework milestones

```
HW1 — Foundations          In Progress   — assignments/hw1-foundations/
HW2 — Systems              In Progress   — assignments/hw2-systems/
HW3 — Data + Scaling       Not Started
HW4 — Post-Training        Not Started
Final Project              Not Started
```

### Final-project checkpoints

```
W8  Proposal               Not Started
W10 Mid-report             Not Started
W16 Paper + Live Demo      Not Started
```

### Git tags

```
v0.1-hw1-foundations                  will be tagged at HW1 submission
v0.2-hw2-systems                      will be tagged at HW2 submission
v0.3-hw3-data-scaling                 will be tagged at HW3 submission
v0.4-post-training                    will be tagged at HW4 submission (covers W8–W11)
v1.0-from-token-to-agent              will be tagged at Final submission
```

## What "Done" means here

For a week to be marked **Done**, all of:

1. `weeks/wXX-*/README.md` answers: What did I learn? What did I build? What did I measure? What did I conclude?
2. Code that belongs in `src/token_to_agent/` is there, not duplicated under the week's folder.
3. Any actual run has an `experiments/wXX/exp-NNN-name/` folder with `config.yaml`, `metadata.json`, `metrics.json`, and a short `README.md`.
4. The week's Pass Criteria from `COURSE.md §2` are met, or the deviation is documented.

For a homework (HW1–HW4, Final) to be tagged, all of:

1. The corresponding weeks are Done.
2. `assignments/hwN/report.md` integrates the weeks' evidence (no copied code, only references to `src/` and `experiments/`).
3. A `v0.X-hwN` tag points at the commit.

## Current known limitations

- **No GPU on the dev host** (Jetson Nano). Triton kernels and CUDA-only paths fall back to `reference_attention.py` at runtime with a logged warning; numerical correctness tests still pass.
- **HW1 / HW2 evidence is "smoke only"** — toy-scale runs that confirm the pipeline works end-to-end, not the full W3 scaling pilot or the full W4 multi-GPU benchmark.
- **W1 char / word tokenizer not yet implemented** — only BPE exists in `src/token_to_agent/tokenizer/bpe.py`. The W1 cross-comparison requires the other two tokenizers; this is the next concrete step.
- **HW1 report and HW2 report were rewritten into `weeks/` view** during the 2026-09-17 restructure. Previous toy numbers preserved under `experiments/w03/exp-001-toy-baseline/` and `experiments/w04/exp-001-triton-vs-pytorch-vs-reference/`.

## Next concrete steps (in order)

1. Implement `src/token_to_agent/tokenizer/char.py` and `src/token_to_agent/tokenizer/word.py`. Add comparison runner + tests.
2. Implement `src/token_to_agent/model/layernorm.py` for the W2 RMSNorm vs LayerNorm ablation.
3. Add `experiments/w02/exp-001-layernorm-vs-rmsnorm/` and `experiments/w02/exp-002-mha-vs-gqa/` ablations.
4. Add `experiments/w03/exp-002-scaling-pilot-20m/` etc. — W3 model-side scaling.
5. After HW1 reaches Pass Criteria, tag `v0.1-hw1-foundations`.