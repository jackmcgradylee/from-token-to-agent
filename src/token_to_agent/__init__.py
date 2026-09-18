"""LLM from token to agent — root package.

Mirrors the layout declared in COURSE.md §23:
    from_scratch/   W1-W4 + W10 DPO/RLVR core — Tokenizer, Transformer, Training,
                   Kernels, DPO loss, RLVR teaching loop. All hand-written.
    systems/        W4-W5 — Distributed (DDP/FSDP/Megatron wrappers), Serving
                   (vLLM/SGLang client + Load Generator), Profiling.
    data/           W6-W7 — Raw pipeline + Synthetic Data Factory.
    post_training/  W8-W11 — SFT/DPO/RLVR scale-train runners (use TRL/slime/verl).
    agent/          W12-W16 — Harness, Environments, Tools, Memory, Self-Eval.
    evaluation/     HW4 + FINAL — Unified eval harness.

The two-track model strategy (COURSE §13) lives here:
    - Scratch Track (~100M): experiments under from_scratch/ + data/
    - Post-Training Track (1.5B-9B): experiments under post_training/
"""

__version__ = "0.5.0"