# W5 — Economics of Inference

## What this week is about

Two questions:

1. **How fast?** TTFT, TPOT, throughput — driven by prefill/decode cost,
   KV-cache memory pressure, continuous batching, and the GPU kernel
   (W4) underneath.
2. **How expensive?** `$ / 1M input tokens`, `$ / 1M output tokens`,
   `$ / successful task` — driven by throughput, GPU $/hr, and utilisation.

Together these answer HW2's submission question: *"How Fast? + How
Expensive?"*. W5 is the systems half of HW2; W4 is the kernel half.

## Status

**Done on the CPU-runnable parts.** The 9 concepts are taught in
`docs/w5-inference-economics.md`. The scheduler logic
(`continuous batching + KV cache accounting + chunked prefill`) is
implemented as a CPU-runnable discrete-event simulator
(`src/token_to_agent/systems/serving/simulated_engine.py`). The KV
cache math is parameterised over `ModelConfig` presets
(Llama-2-7B, Llama-2-70B, Mistral-7B, Phi-3-mini) and matches the
formulas vLLM and SGLang publish. The serving benchmark
(`experiments/w05/exp-003-serving-bench`) sweeps 18 cells in 1.6 s on
CPU and emits CSV / JSON / Markdown artifacts.

**CUDA-required parts deferred.** Real vLLM / SGLang launch, real
FlashAttention-2 decode step, real NCCL tensor-parallel serving — all
gated behind a CUDA host. See `experiments/w05/README-JETSON-LIMITATIONS.md`.

## What was implemented

### Continuous-batching scheduler (CPU-runnable)

File: `src/token_to_agent/systems/serving/simulated_engine.py`.

- Discrete-event simulation on a min-heap of `(time, kind, request_id)`.
- One `decode_step` event per scheduler tick: admit waiting requests
  whose KV cache fits in the budget, advance prefill one chunk, emit
  one decode token per active request.
- KV cache accounting: each request reserves
  `prompt_len + max_output` worth of bytes at admission; freed on
  completion.
- Admission policy: requests that don't fit get `error="kv_oom"`
  (and `n_errors` in the metrics). Requests longer than the model's
  `max_seq_len` get `error="seq_too_long"`.
- Chunked prefill: long prompts are processed in
  `prefill_chunk_tokens` chunks that overlap with decode steps.
- Output tokens per active sequence are parameterised via
  `decode_tps_per_seq` (single knob — calibrate on real GPU).

### KV cache math

File: `src/token_to_agent/systems/inference/kv_cache.py`.

- `ModelConfig` dataclass with presets: `llama2-7b`, `llama2-70b`,
  `mistral-7b`, `phi3-mini`.
- `kv_bytes_per_token(cfg, dtype)` returns per-token KV memory;
  `kv_bytes_for_request(cfg, seq_len, dtype)` returns per-request
  memory; `max_concurrent_at_seq_len(cfg, seq_len, gpu_bytes, dtype,
  overhead_gb)` returns how many concurrent requests fit.
- `cost_per_million_tokens($/hr, tps)` for $/1M math.

Verified numbers (run `python -m src.token_to_agent.systems.inference.kv_cache`):

| Model | per_token | 4k context | 32k context |
|---|---:|---:|---:|
| Llama-2-7B (MHA) | 512 KB | 2 GB | — (max 4096) |
| Llama-2-70B (GQA) | 320 KB | 1.25 GB | — |
| Mistral-7B (GQA-8) | 128 KB | 512 MB | 4 GB |
| Phi-3-mini (MHA) | 384 KB | 1.5 GB | — |

### Load generator

File: `src/token_to_agent/systems/serving/load_generator.py`.

- Poisson arrivals (`rps`) and burst arrivals (5× rate for a configurable
  window) — both with deterministic seed.
- Lognormal prompt / output length distributions (mean + cap).
- Deterministic given `seed`.

### Metrics collector

File: `src/token_to_agent/systems/serving/metrics_collector.py`.

- Per-request `RequestRecord`: arrival, first_token, completion times.
- Derived TTFT / TPOT / E2E in milliseconds.
- `AggregateStats` with p50 / p95 / p99 / max / mean.
- `MetricsCollector.summarise()` rolls everything up including
  throughput (input TPS, output TPS, requests/s) and
  `cost_usd_per_1m_input / output / successful_task` from a
  `cost_per_hour_usd` knob.

### Serving benchmark sweep

File: `experiments/w05/exp-003-serving-bench/runner.py`.

3 (rps) × 3 (prompt_len) × 2 (gen_len) = 18 cells. Llama-2-7B,
fp16, 60 GB KV budget. Reports TTFT/TPOT/E2E p50/p95, throughput,
`$/1M-token`, completion/preemption/error counts. 1.6 s wall-clock on
the Jetson.

## Reproduction

```bash
# All CPU.
PYTHONPATH=. .venv/bin/python -m pytest tests/                                  # 47+ tests
PYTHONPATH=. .venv/bin/python src/token_to_agent/systems/inference/kv_cache.py   # reference numbers
PYTHONPATH=. .venv/bin/python experiments/w05/exp-003-serving-bench/runner.py    # sweep
```

On a CUDA host, additionally:

```bash
pip install vllm
vllm serve meta-llama/Llama-2-7b-hf --port 8000 --max-model-len 4096
# In another shell:
PYTHONPATH=. .venv/bin/python experiments/w05/exp-003-serving-bench/vllm_client.py \
    --endpoint http://localhost:8000 --n-requests 1000 --rps 32
```

## Sample numbers (CPU sim, H100 reference)

`experiments/w05/exp-003-serving-bench/results/serving.md`:

| rps | prompt | gen | n_completed | preempted | errors | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | out TPS | $/1M in | $/1M out |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.0 | 256 | 64 | 200 | 0 | 0 | 16.7 | 35.6 | 20.0 | 189.0 | 0.583 | 2.940 |
| 16.0 | 256 | 256 | 200 | 0 | 0 | 16.3 | 37.8 | 20.0 | 2249.3 | 0.196 | 0.247 |
| 64.0 | 1024 | 256 | 123 | 0 | 77 | 63.1 | 151.7 | 20.0 | 2959.8 | 0.023 | 0.188 |
| 64.0 | 2048 | 256 | 79 | 0 | 121 | 120.0 | 254.9 | 20.0 | 1782.4 | 0.012 | 0.312 |

Read carefully:
- **TPOT is constant ~20 ms** by design: the simulator assumes one
  decode step takes `1 / decode_tps_per_seq` regardless of batch size
  (this matches real GPU behaviour — decode is bandwidth-bound and
  parallelisable). Calibrate `decode_tps_per_seq` on a real GPU box.
- **TTFT scales with prompt_len** (chunked prefill overlaps decode).
  The absolute numbers are not real prefill timings on a GPU; the
  *trend* is correct (TTFT ∝ prompt_len).
- **$/1M tokens** assumes the configured `$2.00/hr H100 on-demand`
  price and the configured `decode_tps_per_seq = 50`. Recompute on
  the actual hardware pricing for production estimates.

## What I learned

- **Memory, not FLOPs, is the binding constraint for serving.** A 7B
  model on H100 80 GB holds the model weights (14 GB) + ~32 concurrent
  4k-context requests' KV cache. The maximum throughput is set by
  how many requests fit, not how fast the GPU can do matmuls.
- **Continuous batching is the single biggest serving win.** With
  static batching, a request that's 99 % done blocks the slot until
  its sibling finishes. With continuous batching, the GPU stays full
  every step.
- **GQA saves KV memory, not compute.** Llama-2-7B (MHA) uses 4×
  more KV cache per token than Mistral-7B (GQA-8) at the same
  head_dim, so Mistral serves ~4× more concurrent requests in the
  same GPU memory.
- **TTFT and TPOT have completely different bottlenecks.** TTFT is
  prefill (compute-bound, ∝ prompt_len²); TPOT is decode
  (bandwidth-bound, ∝ 1 / batch_size's memory traffic). Optimising
  one doesn't move the other.
- **CPU-side scheduler validation is real validation.** Even without
  a GPU, "does continuous batching keep the batch full?" and "does
  the admission policy respect the KV budget?" are testable
  questions. The absolute latency numbers come from the GPU; the
  scheduler semantics come from the simulator.

## Pass criteria (from `COURSE.md` §十 W5)

- [x] 9 concepts explained (`docs/w5-inference-economics.md`).
- [x] vLLM-style scheduler implemented and validated
      (`systems/serving/simulated_engine.py`).
- [x] Load generator implemented (Poisson + burst).
- [x] Sweep over concurrency × prompt_len × gen_len, reporting
      TTFT / TPOT / P50 / P95 / tokens/s / throughput / $/1M.
- [x] Cost / 1M input tokens, cost / 1M output tokens, cost /
      successful task — all reported.
- [ ] **CUDA-host vLLM / SGLang launch** (deferred to
      `experiments/w05/README-JETSON-LIMITATIONS.md`).
- [ ] **Real FlashAttention-2 decode timing** (deferred; calibrated
      `decode_tps_per_seq` knob until a GPU box is available).

## See also

- [`docs/w5-inference-economics.md`](../../docs/w5-inference-economics.md)
  — W5 textbook deliverable.
- [`experiments/w05/exp-003-serving-bench/results/serving.md`](../../experiments/w05/exp-003-serving-bench/results/serving.md)
  — sweep output table.
- [`experiments/w05/README-JETSON-LIMITATIONS.md`](../../experiments/w05/README-JETSON-LIMITATIONS.md)
  — what is gated on CUDA + how to run it.
- [`assignments/hw2-systems/README.md`](../../assignments/hw2-systems/README.md)
  — HW2 views W4 + W5 together as the systems submission.
