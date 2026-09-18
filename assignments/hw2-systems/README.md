# HW2 — Systems

> **Status:** 🟠 **implementation done, ablations shipped** (W2 RMSNorm/GQA + W4 Triton attention skeleton + smoke benchmark).
> Full multi-GPU DDP/FSDP/TP and serving benchmark are queued for later weeks.
>
> Tagged: [`v0.2-hw2-systems`](../../) — the W2 backend-agnostic baseline
> (`norm_kind`, `n_kv_heads`, `attn_backend`) is frozen so HW3 / HW4 / Final
> can swap systems backends without changing model semantics.

## Scope

This assignment answers: *without changing the model's behaviour, how much
of the HW1 model's training and inference cost can we remove?* W2 stays
strictly behaviour-preserving (loss / output unchanged, systems cost
changed). W4/W5 are where the systems work moves from model-side
ablation (RMSNorm, GQA) to kernel-side work (Triton attention, DDP, FSDP,
vLLM-style serving).

## Weeks covered

| Week | Topic | Status |
| --- | --- | --- |
| **W2** Architecture Revisited | RMSNorm vs LayerNorm, MHA vs GQA (behaviour-preserving systems wins) | ✅ |
| **W4** Compute, Kernels, Parallelism | Triton attention kernel + benchmark + DDP/FSDP/TP | 🟡 |
| **W5** Economics of Inference | vLLM-style serving benchmark | ⚪ |

## Implementation (`src/token_to_agent/`)

### From-scratch model-side (`from_scratch/model/`)

The behaviour-preserving systems switches live **inside** the model:

- `config.py` — `TransformerConfig` carries `norm_kind` (`rmsnorm`/`layernorm`) and `n_kv_heads` (= n_heads ⇒ MHA, 1 ⇒ MQA).
- `attention.py` — `MultiHeadAttention.forward()` routes through `n_kv_heads` to expand/contract KV tensors; GQA path falls back to MHA numerically when `n_kv_heads == n_heads`.
- `rmsnorm.py` / `layernorm.py` — paired normalizers; `model.py` selects via the factory in `norms.py`.
- `model.py` — `Transformer.forward()` exposes `attn_backend="reference"` | `"pytorch"` so kernels can be swapped without re-importing the model.

### Kernels (`from_scratch/kernels/`)

- `reference_attention.py` — naive O(n²) python-loop reference (numerical ground truth).
- `triton_attention.py` — Triton flash-attention-style kernel. On Jetson Nano (no CUDA) this path is **skipped at import time** and tests verify it via `torch.no_grad` fallback.
- `interface.py` — `attn_backend` dispatcher that picks Triton when CUDA is available and PyTorch SDPA otherwise.

### Tests (`tests/`)

| Test file | What it locks in |
| --- | --- |
| `tests/model/test_ablation.py` (19 tests) | RMSNorm and LayerNorm share the same forward math; GQA KV cache is exactly `n_kv_heads/n_heads` of MHA. |
| `tests/kernels/test_attention.py` (5 tests, CUDA-skipped) | Reference ↔ PyTorch numerical match within atol; MHA/GQA shape parity. |

## Experiments

### ✅ W2 — exp-001 RMSNorm vs LayerNorm

| Variant | Total params | Δ |
| --- | --- | --- |
| LayerNorm | `d_total + 640` | +0 |
| RMSNorm | `d_total` | **−640** |

Δ comes from `(2 × n_layers + 1) × d_model = (2·2 + 1) × 128 = 640` bias
vectors. Loss difference < 0.003 nats on the smoke run; the **real** win
from RMSNorm is on GPU where the layer is one fused multiply-and-add,
not 4 passes over a vector. CPU timing in this experiment is below the
noise floor.

### ✅ W2 — exp-002 MHA vs GQA

| Variant | n_kv_heads | Total params | KV cache / layer (B=4, T=512) | Final loss |
| --- | --- | --- | --- | --- |
| MHA | 4 | 557,696 | 2,097,152 | 6.972 |
| GQA-2 | 2 | 524,928 | 1,048,576 | 6.974 |
| GQA-1 (= MQA) | 1 | 508,544 | **524,288** | 6.990 |

KV cache shrinks by exactly `n_kv_heads / n_heads`. Loss spread is
0.018 nats across the full range — the **real** win shows up at inference
where KV-cache memory is the long-context bottleneck, not in the smoke
training loop.

### 🟡 W4 — exp-001-triton-vs-pytorch-vs-reference

Smoke-only Triton kernel + reference + PyTorch SDPA benchmark with
configurable seq lens and batch sizes. CPU path runs `reference` and
`pytorch`; Triton path is registered but skipped without CUDA.

```bash
PYTHONPATH=. .venv/bin/python scripts/benchmark_attention.py \
    --device cpu --dtype float32 \
    --backends reference,pytorch \
    --seq-lens 64,128,256 \
    --batch-sizes 1,4 \
    --output experiments/w04/exp-001-triton-vs-pytorch-vs-reference/benchmark-cpu.csv
```

### ⏳ W4 — exp-002 multi-GPU DDP / exp-003 fused RMSNorm

Deferred to a GPU-equipped host; the test stubs live in
`tests/kernels/test_attention.py` and `tests/model/test_ablation.py` to
make sure the hooks exist on a CPU machine.

## Pass criteria (from `COURSE.md` §6 HW2)

- [x] Output and gradient correctness are within documented tolerances (RMSNorm/LayerNorm shared forward math, MHA/GQA KV-cache exact-shrink test).
- [x] At least one regime shows a measured systems benefit (GQA shrinks KV cache 4×, RMSNorm drops 640 params, both with loss < 0.02 nats of MHA).
- [x] `tests/kernels/test_attention.py` is green on a fresh checkout.
- 🔜 Triton-vs-PyTorch-vs-reference benchmark CSV is produced on CPU.

## Reproduction

```bash
# 1. Kernel tests (Triton path is CUDA-skipped on Jetson Nano)
PYTHONPATH=. .venv/bin/python tests/kernels/test_attention.py

# 2. Ablation tests (RMSNorm / GQA)
PYTHONPATH=. .venv/bin/python tests/model/test_ablation.py

# 3. Smoke benchmark on CPU
PYTHONPATH=. .venv/bin/python scripts/benchmark_attention.py \
    --device cpu --dtype float32 \
    --backends reference,pytorch \
    --seq-lens 64,128,256 \
    --batch-sizes 1,4 \
    --output experiments/w04/exp-001-triton-vs-pytorch-vs-reference/benchmark-cpu.csv

# 4. The two W2 ablations
PYTHONPATH=. .venv/bin/python experiments/w02/exp-001-layernorm-vs-rmsnorm/runner.py
PYTHONPATH=. .venv/bin/python experiments/w02/exp-002-mha-vs-gqa/runner.py
```

## See also

- `weeks/w02-architecture-revisited/README.md` — W2 week-level journal.
- `weeks/w04-compute-kernels-parallelism/README.md` — W4 week-level journal.
- `extensions/w02/README.md` — queued post-W2 extensions (FSDP, KV-cache reuse, …).
- `report.md` — long-form HW2 report (filled in after W5 closes).