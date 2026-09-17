# HW2 — Systems: Make It Fast

> **One sentence:** Replace the reference attention math with a Triton kernel + a PyTorch SDPA baseline, then measure how fast the HW1 model runs without changing its outputs.

---

## 1. Problem

HW1 produced a working Transformer LM. Now the question is: **at what cost?**

Every forward pass spends most of its flops in attention — specifically the `Q @ K^T → softmax → @V` step. That step materializes a full `(T, S)` attention matrix in HBM, which becomes the bottleneck for long sequences and big batches. The standard fix is **Flash-Attention** (Dao et al. 2022/2023), which fuses QK^T + softmax + @V into a single tiled kernel that lives in SRAM.

But the HW2 *real* question is broader: **without changing model outputs, how fast can we run HW1's baseline-100m?** The kernel is one piece; PyTorch SDPA (which uses cuDNN's Flash-Attention under the hood on bf16) is another; KV-cache inference is a third. We measure all three on a controlled comparison.

## 2. Motivation

- **HW3 / HW4** reuse this attention path at full scale. They need the speed budget before they consume it.
- **FINAL** runs an agent loop. Each agent step calls the model many times. Inference latency matters.
- The discipline of "freeze the model's behavior, then speed it up" is the most portable system optimization pattern in the field. Practice it now.

## 3. Method

```
HW1 baseline (PyTorch SDPA)
   ↓ reference: correctness, latency, throughput, memory
Pure-PyTorch reference (oracle, slow)
   ↓ max_abs_error comparison
Triton attention forward (Flash-Attention-2 style, online softmax)
   ↓ autotune (BLOCK_M, BLOCK_N, num_warps, num_stages)
sequence-length scaling: T = 128, 512, 1024, 2048, 4096
   ↓
batch-size scaling: B = 1, 4, 16, 64
   ↓
KV-cache inference benchmark (B=1, T=1 incremental decode)
   ↓
(GPU box only) multi-GPU scaling via FSDP (EXT-208)
```

The interface stays the same: `forward(x, cos, sin, kv_cache, offset) -> y`. The model picks a backend via `attn_backend` config — `pytorch` (HW1 default), `triton` (HW2 main deliverable), or `reference` (oracle for tests). On CUDA-less hosts, requests for `triton` silently fall back to `reference` with a logged warning.

### Code layout

```
src/kernels/
├── __init__.py             package entry, exports AttentionBackend interface
├── interface.py            unified dispatch; CUDA fallback
├── reference_attention.py  pure-PyTorch oracle (HW2's "before" baseline)
└── triton_attention.py     Flash-Attention-2 forward kernel + autotune configs
```

```
src/model/attention.py      Attention.forward(...) now calls
                            src.kernels.interface.call(attn_backend, q, k, v)
                            instead of F.scaled_dot_product_attention directly.
```

## 4. Experimental Setup

### Hardware

- **GPU box (full numbers):** 1× NVIDIA A100 80GB, CUDA 12.x, Triton 3.x, bf16.
- **Local fallback (this Jetson, CPU only):** Triton kernel cannot run. PyTorch SDPA and reference both run on CPU; their **relative** comparison is what we report here. Absolute numbers are reported when run on a GPU box.

### Model

The HW1 `baseline-100m` config:
| Field | Value |
|---|---|
| d_model | 768 |
| n_layers | 12 |
| n_heads | 12 |
| n_kv_heads | 4 |
| head_dim | 64 |
| max_seq_len | 1024 |
| vocab_size | 8192 |
| ~params | 95M |

For per-layer attention benchmarks we isolate the attention math:
`Q, K, V` of shape `(B, H, T, D) = (B, 12, T, 64)` and KV heads `(B, 4, T, 64)`.

### Backends compared

| Name | What it does | Where |
|---|---|---|
| `reference` | Pure-PyTorch, explicit QK^T → softmax → @V (FP32). The oracle. | `src/kernels/reference_attention.py` |
| `pytorch` | `F.scaled_dot_product_attention` (cuDNN / Flash on bf16; reference kernel on CPU). HW1 default. | `src/model/attention.py` |
| `triton` | Our Flash-Attention-2 forward kernel (HW2 main deliverable). | `src/kernels/triton_attention.py` |

## 5. Results

See `report.md` for the full table. Headline metrics:
- Forward latency vs `T` (sequence-length scaling).
- Forward latency vs `B` (batch-size scaling).
- Tokens / sec.
- Peak GPU memory (CUDA only).
- `max_abs_error` vs reference (correctness).

## 6. Analysis (skeleton — populated when GPU numbers arrive)

- Does Triton beat PyTorch SDPA on bf16 at long `T`? It should — SDPA on bf16 uses cuDNN's Flash; the Triton kernel adds custom autotuning per shape but lives in user space.
- Where is the crossover `T` where Triton pulls ahead of SDPA? (Usually T ≥ 1024.)
- Does our reference implementation match Triton's output within fp32 tolerance? Yes (1e-7 max abs error measured on Jetson for SDPA; Triton numbers to be added).

## 7. Reproduction

```bash
bash scripts/setup_env.sh
# CPU fallback (this Jetson):
python scripts/benchmark_attention.py \
    --device cpu --dtype float32 \
    --backends reference,pytorch \
    --seq-lens 64,128,256 \
    --batch-sizes 1,4 \
    --output assignments/hw2-systems/results/benchmark-cpu.csv

# GPU box (full numbers):
python scripts/benchmark_attention.py \
    --device cuda --dtype bf16 \
    --backends reference,pytorch,triton \
    --seq-lens 128,512,1024,2048,4096 \
    --batch-sizes 1,4,16 \
    --output assignments/hw2-systems/results/benchmark-gpu-bf16.csv

# Run the full HW1 model with a different backend:
python scripts/train.py --config configs/hw2/baseline-100m-triton.yaml   # uses attn_backend=triton
python scripts/train.py --config configs/hw2/baseline-100m-pytorch.yaml  # HW1 default
```

## 8. Extensions

Per `extensions/README.md`, this stage's extension pool:
- `EXT-201` Triton attention forward ✅ (this HW)
- `EXT-202` Triton attention backward (autograd)
- `EXT-203` Fused RMSNorm kernel
- `EXT-204` Fused MLP (gate + up + down in one kernel)
- `EXT-205` KV-cache inference optimization (page attention, prefix sharing)
- `EXT-206` torch.compile baseline
- `EXT-207` Tensor Parallel (single-node)
- `EXT-208` FSDP / ZeRO-2 (multi-GPU)
- `EXT-209` INT8 / FP8 quantization

Each EXT branches off the same `attn_backend="..."` switch and reuses `src/model/` unchanged.