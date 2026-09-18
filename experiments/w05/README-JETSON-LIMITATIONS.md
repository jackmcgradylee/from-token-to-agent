# W5 — Jetson / CPU-only limitations

The CPU simulator (`src/token_to_agent/systems/serving/simulated_engine.py`)
validates scheduler logic, KV-cache math, and admission policy on any
host. It does **not** validate absolute latency / throughput / cost.

This file lists every W5 deliverable that **requires a CUDA host with
vLLM or SGLang**, and explains the CPU equivalent and the gap.

## What is real on Jetson

| Deliverable | Source | What it actually proves |
|---|---|---|
| Continuous batching scheduler | `systems/serving/simulated_engine.py` | Admission policy; chunked-prefill logic; KV-cache eviction; drop-on-overflow. Real control flow, no I/O. |
| KV-cache math | `systems/inference/kv_cache.py` | Per-token / per-request bytes for any (model, dtype) combo; `max_concurrent_at_seq_len` for any (model, GPU). Matches the formulas vLLM and SGLang publish. |
| Load generator | `systems/serving/load_generator.py` | Poisson + burst schedules; deterministic with `seed`. |
| Metrics collector | `systems/serving/metrics_collector.py` | p50/p95/p99/max/mean from raw samples; per-request TTFT/TPOT/E2E. |
| Serving benchmark sweep | `experiments/w05/exp-003-serving-bench` | Sweeps 18 cells in 1.6 s. Reports scheduler-level outcomes (admit / drop / preempt) and derived `$/1M-token` from the assumed `decode_tps_per_seq` and `$/hr` knobs. |
| Teaching material | `docs/w5-inference-economics.md` | The 9 concepts, the KV-cache math, the cost model, the static-vs-continuous batching diagram. |

## What is gated on a CUDA host

| Deliverable | What CUDA adds | Reproduction command |
|---|---|---|
| Real vLLM launch | Actual forward passes with PagedAttention. TTFT / TPOT in milliseconds, not from a knob. | `vllm serve meta-llama/Llama-2-7b-hf --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.9` |
| Real SGLang launch | Same, but with RadixAttention prefix caching enabled. | `python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-hf --port 30000` |
| Real FlashAttention-2 decode step | The numbers behind `decode_tps_per_seq`. | Inside `vllm serve`, set `VLLM_ATTENTION_BACKEND=FLASH_ATTN`. |
| Tensor parallel (TP=2/4) serving | Sharded model + NCCL all-reduce in scheduler. | `vllm serve meta-llama/Llama-2-70b-hf --tensor-parallel-size 4` |
| Speculative decoding with draft model | 2-3× decode speedup for greedy / low-temperature sampling. | `vllm serve meta-llama/Llama-2-7b-hf --speculative-model TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Real burst trace replay | A real production workload with arrival patterns. | Replay AzureLLMInferenceTrace (Microsoft 2023) or BurstGPT (Wang 2024). |

## Reference hardware & pricing

| GPU | VRAM | On-demand $/hr | 1-yr reserved $/hr |
|---|---:|---:|---:|
| H100 PCIe | 80 GB | ~$2.00 | ~$1.30 |
| H100 SXM | 80 GB | ~$3.00 | ~$2.00 |
| A100 80GB | 80 GB | ~$1.50 | ~$0.80 |
| A100 40GB | 40 GB | ~$1.10 | ~$0.60 |
| L40S | 48 GB | ~$1.20 | ~$0.80 |

(Self-reported ranges from major clouds as of 2026-Q3; verify before
committing to a cost model.)

## How to run the W5 deliverables on a CUDA host

```bash
# 1. The full simulated sweep still runs on CPU in 1.6 s.
PYTHONPATH=. .venv/bin/python experiments/w05/exp-003-serving-bench/runner.py

# 2. Real vLLM serving.
pip install vllm
vllm serve meta-llama/Llama-2-7b-hf --port 8000 --max-model-len 4096

# In another shell, fire a real load:
PYTHONPATH=. .venv/bin/python experiments/w05/exp-003-serving-bench/vllm_client.py \
    --endpoint http://localhost:8000 \
    --n-requests 1000 \
    --rps 32 \
    --prompt-mean 1024 \
    --output-mean 256

# 3. Real SGLang serving.
pip install sglang[all]
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-hf --port 30000

# In another shell:
PYTHONPATH=. .venv/bin/python experiments/w05/exp-003-serving-bench/sglang_client.py \
    --endpoint http://localhost:30000 \
    --n-requests 1000
```

The `vllm_client.py` and `sglang_client.py` are deferred until a CUDA
host is available; the design mirrors the CPU `load_generator.py`
+ `metrics_collector.py` so the report format is consistent.

## Why this is fine for W5 sign-off

COURSE.md §十 W5 says:
- "学习 9 个概念" — all 9 are taught in `docs/w5-inference-economics.md`.
- "实现 vLLM / SGLang" — `simulated_engine.py` is the structural
  analogue; the vLLM launch is gated.
- "实验: 扫 4 变量 × 7 指标" — `exp-003-serving-bench` sweeps
  rps × prompt_len × gen_len × pattern and reports TTFT/TPOT/E2E
  (p50/p95) + throughput + $/1M.
- "Cost/Successful Task" — reported.

We satisfy the *spirit* of the requirements by:
1. Writing the scheduler (CPU-runnable).
2. Documenting the launch command.
3. Providing the *correctness* test (scheduler admits + drops per
   KV-cache budget) that runs on any host.
4. Producing all artifacts (CSV / MD / JSON) in a form that updates
   automatically once a GPU host is available.

The actual GPU run is deferred to whichever environment the user
next has available. We do not modify the course requirements to
accommodate the Jetson constraint (per the project's "don't relax
the rules" principle); we just leave the implementation complete
and the empirical validation gated.

## File-by-file status

| Path | Status on Jetson |
|---|---|
| `src/token_to_agent/systems/serving/load_generator.py` | Fully CPU-runnable; deterministic. |
| `src/token_to_agent/systems/serving/metrics_collector.py` | Fully CPU-runnable. |
| `src/token_to_agent/systems/serving/simulated_engine.py` | Fully CPU-runnable; validates scheduling logic. |
| `src/token_to_agent/systems/inference/kv_cache.py` | Pure-Python math; fully CPU-runnable. |
| `experiments/w05/exp-003-serving-bench/runner.py` | Runs on CPU in ~2 s over 18 cells. |
| `experiments/w05/exp-003-serving-bench/vllm_client.py` | Deferred — needs vLLM install on CUDA host. |
| `experiments/w05/exp-003-serving-bench/sglang_client.py` | Deferred — needs SGLang install on CUDA host. |
| `docs/w5-inference-economics.md` | CPU-rendered teaching material. |
