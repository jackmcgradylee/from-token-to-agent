# HW2 — Systems · Report

> **Status: In Progress** — smoke verified only. Not yet tagged.

## 1. Problem

Write a Triton attention kernel and measure how fast the HW1 model runs without changing its outputs.

## 2. Motivation

HW3 / HW4 / FINAL all reuse this attention path. The Triton vs SDPA vs reference comparison quantifies where the speedup comes from and what's left on the table.

## 3. Method

See the two week folders:
- `weeks/w04-compute-kernels-parallelism/README.md`
- `weeks/w05-economics-of-inference/README.md`

## 4. Experimental setup

- **Hardware (smoke):** Jetson Nano, ARMv8, 3.9 GB RAM, no CUDA.
- **Shape:** `(B, H=12, H_kv=4, T, D=64)` — the attention math layer of baseline-100m.
- **Backends:** reference (pure-PyTorch, oracle), pytorch (F.scaled_dot_product_attention), triton (custom Flash-Attention-2 forward kernel).

## 5. Results (smoke only — Jetson CPU fp32)

| B | T | reference (ms) | pytorch (ms) | speedup | max abs err |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 5.484 | 1.671 | **3.28×** | 1.49e-7 |
| 1 | 128 | 8.487 | 4.048 | **2.10×** | 1.19e-7 |
| 1 | 256 | 26.404 | 13.138 | **2.01×** | 1.34e-7 |
| 4 | 64 | 18.299 | 4.415 | **4.14×** | 1.19e-7 |
| 4 | 128 | 28.975 | 19.818 | **1.46×** | 1.34e-7 |
| 4 | 256 | 83.408 | 49.585 | **1.68×** | 1.34e-7 |

Triton row **skipped** on Jetson (no CUDA). GPU-box numbers pending.

## 6. Analysis (preliminary)

- PyTorch SDPA already beats the pure-PyTorch reference by 1.5–4× on CPU fp32. The reference implementation is a faithful oracle (max abs err 1e-7).
- Speedup drops as T grows (memory-bound on CPU). This is the regime where GPU + Triton should pull ahead.
- The Triton kernel correctness must be re-validated on a GPU box.

## 7. Reproduction

See `assignments/hw2-systems/README.md` § Reproduction.

## 8. Extensions

- `EXT-W4-01` Fused RMSNorm kernel
- `EXT-W4-02` Triton attention backward
- `EXT-W4-03` Fused SwiGLU
- `EXT-W4-04` Sequence parallelism
- `EXT-W5-01` Prefix cache economics
- `EXT-W5-02` Speculative decoding cost analysis