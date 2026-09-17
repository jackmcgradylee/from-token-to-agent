# HW2 — Systems

> **Status:** **In Progress** (smoke only — Jetson CPU fp32). Triton kernel + reference + PyTorch SDPA benchmark implemented and run. **Full W4+W5 deliverables** (multi-GPU DDP/FSDP/TP, seq sweep 512/2K/8K, forward + backward latency, gradient correctness, vLLM/SGLang serving benchmark) are not yet done. This HW will be tagged `v0.2-hw2-systems` only after all pass criteria are met.

## Weeks covered

- **W4 — Compute, Kernels, Parallelism:** `weeks/w04-compute-kernels-parallelism/`
- **W5 — Economics of Inference:** `weeks/w05-economics-of-inference/`

## Implementation

- **Triton kernel (W4):** `src/token_to_agent/kernels/triton_attention.py`
- **Reference oracle:** `src/token_to_agent/kernels/reference_attention.py`
- **Backend dispatch:** `src/token_to_agent/kernels/interface.py`
- **Model integration:** `src/token_to_agent/model/attention.py` + `src/token_to_agent/model/model.py` (`attn_backend` config)
- **Serving (W5):** `src/token_to_agent/serving/` ⏳

## Experiments

- `experiments/w04/exp-001-triton-vs-pytorch-vs-reference/` ✅ smoke only
- `experiments/w04/exp-002-multi-gpu-ddp/` ⏳
- `experiments/w04/exp-003-fused-rmsnorm/` ⏳
- `experiments/w05/exp-001-serving-benchmark/` ⏳

## Pass criteria (from COURSE.md §2 HW2)

- output and gradient correctness are within documented tolerances
- at least one regime shows a measured systems benefit or a clear explanation of why it does not

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_attention.py \
    --device cpu --dtype float32 \
    --backends reference,pytorch \
    --seq-lens 64,128,256 \
    --batch-sizes 1,4 \
    --output experiments/w04/exp-001-triton-vs-pytorch-vs-reference/benchmark-cpu.csv

PYTHONPATH=src .venv/bin/python tests/kernels/test_attention.py
```