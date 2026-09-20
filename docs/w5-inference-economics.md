# W5 — Economics of Inference (textbook deliverable)

This document is the W5 reading material: the *why* behind serving
economics. It pairs with the source code in
`src/token_to_agent/systems/serving/` and
`src/token_to_agent/systems/inference/`, plus the experiment in
`experiments/w05/exp-003-serving-bench/`.

## 1. Why is inference different from training?

Training is **arithmetic-bound** and **bandwidth-comfortable**: you do
O(6ND) FLOPs in one big forward+backward, hit the GPU's compute peak,
and finish in hours/days. Memory is dominated by parameters + optimiser
state + activations, all of which can be checkpointed and sharded.

Serving is **latency-bound** and **memory-hostile**:

- A user is waiting on the response. The metric is **TTFT** (time to
  first token) and **TPOT** (time per output token), not total FLOPs.
- Requests arrive **one at a time, irregularly**. The hardware is
  either idle or saturated; you cannot "batch the night".
- The KV cache grows **per request** and per generated token. Without
  clever scheduling, your GPU's HBM runs out long before your GPU's
  FLOPs do.

A 7B model in fp16 is ~14 GB of weights. On an H100 (80 GB), you have
~65 GB left for KV cache + activations + system overhead. With static
batching, that's room for **maybe 8 concurrent requests at 4k context**.
With continuous batching and prefix caching, you can get **64+**.

The difference is not a small engineering detail. It is the difference
between "we can serve the launch traffic" and "we cannot".

## 2. The 9 concepts

### 2.1 Prefill

The phase where the engine processes the input prompt **all at once**,
building the KV cache for every transformer layer. Prefill is
**compute-bound** (the matmul is `B=1 × H × T × D × T` = O(T²) FLOPs)
and produces a fully-warmed KV cache.

Prefill latency is dominated by the prompt length squared: a 4k prompt
takes ~16× longer than a 1k prompt, all else equal.

### 2.2 Decode

The phase where the engine generates one token at a time, autoregressively.
At each step:
1. Forward pass on the new token only (1× D inputs to matmul, not T×D).
2. Append the new (K, V) to the cache.
3. Sample the next token.

Decode is **memory-bandwidth-bound**: each step touches ~14 GB of weights
(for 7B fp16) and writes a few KB to KV cache. The FLOPs per step are
small; the bottleneck is reading the weights from HBM.

This is why **TTFT** (prefill latency) and **TPOT** (decode latency) are
measured separately — they have completely different bottlenecks.

### 2.3 TTFT (Time To First Token)

`TTFT = end_of_prefill - request_arrival`. The user is staring at a
loading spinner until this completes. TTFT is dominated by prefill.

SLO: typical chat products target TTFT ≤ 300 ms for short prompts,
≤ 1.5 s for 4k-token prompts.

### 2.4 TPOT (Time Per Output Token)

`TPOT = (end_of_generation - end_of_prefill) / num_output_tokens`.
Average time to generate one token after the first one is out.

SLO: typical chat products target TPOT ≤ 30 ms (so 33 tokens/s for
just the decode phase). Real end-to-end TPS = 1 / (TTFT/N + TPOT) for
N output tokens.

### 2.5 Continuous Batching

The scheduler is allowed to insert new requests into a "decode batch"
**at every decode step**, not just at request boundaries. A request
that finishes leaves the batch immediately; a request that is still
generating keeps its slot.

This is **the** big serving win. Static batching (all requests must
finish together) wastes GPU cycles on padding; continuous batching
keeps every FLOP earning its keep.

Reference: vLLM's PagedAttention paper (Kwon et al., SOSP 2023) shows
23× throughput vs naive static batching at high concurrency.

### 2.6 KV Cache

For each transformer layer, each request holds:

```
K_i ∈ ℝ^(H_kv × T_i × D)
V_i ∈ ℝ^(H_kv × T_i × D)
```

per layer, where T_i is the current sequence length for request i.
Two reference numbers:

**Mistral-7B** (GQA, n_kv_heads=8, head_dim=128, 32 layers, fp16):

```
KV per token per layer = 2 × 8 × 128 × 2 = 4096 bytes = 4 KB
KV per token total     = 32 × 4 KB = 128 KB / token
```

So a single 4k-context request costs **512 MB of KV cache**.

**Llama-2-7B** (MHA, n_kv_heads=32, head_dim=128, 32 layers, fp16):

```
KV per token per layer = 2 × 32 × 128 × 2 = 16 KB
KV per token total     = 32 × 16 KB = 512 KB / token
```

So a single 4k-context request costs **2 GB of KV cache**. Roughly
4× the Mistral number because GQA saves KV memory 4×.

Concrete budget on H100 80GB with 2 GB activation overhead, fp16:

| Model | 4k concurrent budget |
|---|---:|
| Llama-2-7B (MHA) | ~32 requests |
| Mistral-7B (GQA-8) | ~129 requests |

Both leave ≤ 1 GB headroom — **memory is the binding constraint for
serving**, not compute.

See `src/token_to_agent/systems/inference/kv_cache.py` for the
parameterised version (Llama-2-7B, Llama-2-70B, Mistral-7B, Phi-3-mini
presets).

### 2.7 Prefix Cache (a.k.a. Automatic Prefix Caching)

Many real requests share a long system prompt ("You are a helpful
assistant. The following is a conversation about legal contracts...").
With prefix caching, the engine hashes prompt prefixes and reuses the
KV cache for matching blocks.

Effect: a request with a 2k-token shared prefix pays the prefill cost
of only the **delta** (the new tokens), not the prefix. In heavy
multi-turn chat traffic this can save **50-80%** of prefill FLOPs.

Reference: SGLang's RadixAttention (Zheng et al., 2024).

### 2.8 Speculative Decoding

A small **draft model** proposes N candidate tokens; the large **target
model** verifies all N in **one forward pass** (because the verification
forward is parallel across N positions). If the draft is right, you
generated N tokens at the cost of ~1 target forward.

Effect: 2-3× wall-clock speedup for greedy decoding, less for sampling.
Reference: Leviathan et al. 2023, Chen et al. 2023.

Worth knowing but not the dominant lever — **continuous batching + KV
cache management** are usually the bigger wins.

### 2.9 Serving Scheduler

The component that decides **which requests are in the next decode
batch**, **how memory is allocated**, and **what to do when KV cache
is full** (preempt? swap to CPU? reject?).

Three families:
- **FCFS / first-come-first-served**: simple, can starve.
- **Priority + aging**: production-grade, e.g. prefill-priority with
  decode batching.
- **Chunked prefill + decode merging** (vLLM v0.4+, SGLang v0.2+):
  long prompts are split into chunks that interleave with decode
  steps, eliminating the prefill-decode "cliff".

The W5 simulated engine implements FCFS continuous batching plus a
basic chunked-prefill shim.

## 3. KV Cache math (H100 80 GB reference)

Let:
- `L` = number of transformer layers (32 for 7B)
- `H_kv` = number of KV heads (8 with GQA on Llama-2-7B)
- `D` = head dim (128)
- `T_max` = max sequence length (4096)
- `dtype_bytes` = 2 for fp16 / bf16, 1 for int8, 0.5 for int4
- `B` = concurrent requests

```
KV_bytes_per_token_per_layer = 2 × H_kv × D × dtype_bytes
KV_bytes_per_token           = L × KV_bytes_per_token_per_layer
KV_bytes_per_request_max      = T_max × KV_bytes_per_token
Total_KV_at_full_concurrency = B × KV_bytes_per_request_max
```

Example (Llama-2-7B, fp16, T_max = 4096):

```
per token per layer = 2 × 8 × 128 × 2 = 4096 bytes = 4 KB
per token total     = 32 × 4 KB = 128 KB
per request max     = 4096 × 128 KB = 512 MB
```

So:
- 16 concurrent requests at 4k context = 8 GB KV cache.
- 64 concurrent requests at 4k context = 32 GB KV cache.
- 128 concurrent requests at 4k context = 64 GB KV cache.

The 65 GB free H100 budget eats 64 GB at 128 requests, leaving ~1 GB
for activations and overhead — tight. **Memory, not compute, is the
binding constraint for serving.**

For W5's simulated engine we parameterise the model and let the
scheduler refuse when KV cache would overflow.

## 4. Continuous batching vs static batching

Static batching (think: HuggingFace `pipeline` with `batch_size=N`):

```
req 1: |---prefill---|--decode--decode--decode--decode--|
req 2:                |---prefill---|--decode--decode--decode--decode--decode--|
req 3:                                |---prefill---|--decode--decode--|
req 4:                                                |---prefill---|--decode--decode--|
batch:                                              [####][####][####][####]
                                                                  ^ idle slots
```

Continuous batching:

```
req 1: |---prefill---|--d--d--d--d--d--d--d--|
req 2:                |---prefill---|--d--d--d--d--d--d--d--|
req 3:                                |---prefill---|--d--d--d--d--|
req 4:                                                |---prefill---|--d--d--d--|
batch:                                              [d d d d] all active every step
```

The GPU is always full. The cost is **scheduler complexity**: you must
manage per-request KV cache as sequences grow and shrink independently.

## 5. Cost model

A 7B model on H100 PCIe ($2/hr on-demand, $1.30/hr 1-yr reserved):

```
peak FLOPs           = 1.5e15 FP16 = 1500 TFLOPS
sustained utilisation= 40% (typical for serving: mix of prefill + decode)
sustained TFLOPS     = 600
throughput (decode)  = 600 / model_flops_per_token
                       = 600 / (2 × 7e9)  ≈ 43,000 tokens/s aggregate
```

(Note: this is **decode only**. Prefill is heavier per token.)

```
$/1M_tokens = $/hr × (1e6 / TPS) / 3600
            = 2.00  ×  1e6 / 43000  /  3600
            ≈ $0.013 / 1M output tokens
```

For comparison:
- OpenAI gpt-4o-mini: $0.60 / 1M output tokens (Oct 2024 pricing).
- Self-hosted 7B on H100: ~$0.013 / 1M output tokens (assumptions
  above).

The headline ratio is 45×. Real numbers are 10-30× because (a)
self-hosted has fixed-cost amortisation issues, (b) bursty traffic
makes sustained utilisation hard, (c) you're paying for the GPU even
when idle.

`experiments/w05/exp-003-serving-bench/` parameterises this over
{H100, A100 80GB, A100 40GB} × {on-demand, 1-yr reserved} × {40%, 70%
utilisation} so you can see how the cost curve bends.

## 6. What W5 measures

| Metric | Definition |
|---|---|
| `TTFT_p50_ms` | Median time from request arrival to first token. |
| `TTFT_p95_ms` | 95th percentile TTFT. |
| `TPOT_p50_ms` | Median time per output token, post-first-token. |
| `TPOT_p95_ms` | 95th percentile TPOT. |
| `e2e_p50_ms` | End-to-end latency (arrival → last token). |
| `e2e_p95_ms` | 95th percentile e2e. |
| `throughput_tps` | Aggregate tokens/s across all in-flight requests. |
| `throughput_rps` | Requests/s the engine sustained. |
| `kv_cache_peak_gb` | Peak KV cache memory used. |
| `$/1m_input` | Cost per 1M input tokens at the configured $/hr. |
| `$/1m_output` | Cost per 1M output tokens. |
| `$/successful_task` | Cost per successful request. |

The `simulated_engine.py` mocks the model forward with a configurable
"tokens/s per request" knob, which lets us sweep the parameter space
without actually serving tokens.

## 7. Why this is CPU-runnable (and what isn't)

`experiments/w05/README-DEV-HOST-LIMITATIONS.md` lists every CUDA-gated
deliverable. The short version:

- **CPU-runnable**: load generator, metrics collector, KV cache math,
  continuous batching scheduler (operating on real tensors, just not
  transformer-forward), cost-model sweep, all the 9-concept teaching
  material.
- **CUDA-gated**: actual vLLM / SGLang launch, real prefill timing on
  a 7B model, real FlashAttention decode step, real NCCL tensor-parallel
  serving.

The CPU simulated engine **validates the scheduler logic** — does
continuous batching actually keep the batch full? Does KV cache
overflow behave? Does TTFT distribution shift under burst load? — but
it does **not** validate the absolute numbers (tokens/s, $/1M). Those
need a GPU host.

## 8. Connection to W4

The W4 Triton attention kernel is what makes decode fast in the first
place. KV cache + Triton decode = the reason a 7B model can sustain
40-60 tokens/s on one H100. Without the kernel, the GPU is bandwidth-
starved and you can't hit those numbers.

So:
- W4 = "make one forward fast".
- W5 = "use that speed to serve many users, cheaply".

Together they answer **How Fast?** (W4 TPS) and **How Expensive?**
(W5 $/token) — which is HW2's submission.

## 9. Open TODOs gated on CUDA

- Real vLLM launch with `vllm serve meta-llama/Llama-2-7b-hf`.
- Real SGLang launch with `python -m sglang.launch_server`.
- Tensor-parallel serving on 2/4 GPUs.
- Speculative decoding with a draft model (e.g. Llama-2-7B + TinyLlama).
- Production trace replay (Azure LLM serving trace, MOSAICML trace).

See `experiments/w05/README-DEV-HOST-LIMITATIONS.md` for the exact
launch commands.
