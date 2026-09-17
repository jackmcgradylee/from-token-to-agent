# W5 — notes.md

## The TTFT / TPOT decomposition

**TTFT (Time To First Token)** = prefill latency. Dominated by compute (parallel over the prompt).

**TPOT (Time Per Output Token)** = decode latency. Dominated by memory bandwidth (one token at a time, but you re-read the entire KV cache).

Different scaling laws:

- TTFT ~ `prompt_len × d_model²` (compute-bound)
- TPOT ~ `total_seq_len × d_model² / mem_bandwidth` (memory-bound)

A 4× faster kernel might cut TTFT 4× but only cut TPOT 2×. Reporting only "speedup" hides this.

## Why "faster ≠ cheaper"

The W4 Triton kernel cuts attention compute. But serving has other costs:

- KV cache memory (grows linearly with concurrent requests × prompt + gen length)
- Scheduling overhead (continuous batching)
- Prefix cache hit rate (if you serve the same system prompt to many requests, you can reuse its KV)
- Speculative decoding acceptance rate (draft model cost vs. verification savings)

The interesting W5 question is: **for a given task type, which regime is cheapest?** Batch-of-1 low-latency? High-concurrency throughput? Prefix-heavy chatbot?

## Reading

- `docs/reading/`
- `COURSE.md §2 W5`

## Open questions

- Speculative decoding for 0.1B is overkill; for 1.5B+ it's where the interesting economics are.
- Continuous batching: how do you measure it without a real serving framework?