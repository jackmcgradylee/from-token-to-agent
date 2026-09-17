"""LLM from-scratch to agent — root package.

This package is the growing codebase. Every HW writes into a sub-package here:
  tokenizer/   HW1 — BPE
  model/       HW1 — Transformer LM
  training/    HW1 — training loop, generation
  data/        HW1 dataset reader (HW3 will extend)
  kernels/     HW2 — Triton attention etc.
  post_training/ HW4 — SFT/DPO/RLVR
  environments/ FINAL — verifiable envs
  harness/     FINAL — agent loop
  evaluation/  HW4 + FINAL — unified eval
"""

__version__ = "0.1.0"