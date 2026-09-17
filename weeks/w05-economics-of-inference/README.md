# W5 — Economics of Inference

## What this week is about

When a model becomes faster, does serving it actually become cheaper and more useful?

## Status

Not Started. The W4 Triton kernel speedup is the foundation; W5 needs a serving benchmark (vLLM / SGLang) and cost-per-task measurement.

## To be implemented

- [ ] `src/token_to_agent/serving/` — load-generator + metrics-collector scaffolding.
- [ ] Integration with vLLM or SGLang for the baseline 0.1B (and ideally the 1.5B–9B base on a GPU box).
- [ ] Sweep: concurrency, batch size, prompt length, generation length, prefix cache on/off.
- [ ] Record: TTFT, TPOT, p50/p95 latency, tokens/s, GPU memory, request throughput, $ / 1M tokens.

## Pass criteria (from COURSE.md §2 W5)

- [ ] Service runs against real requests.
- [ ] Cost-per-successful-task reported, not just latency.

## Open questions

- vLLM and SGLang both target frontier models (≥ 7B). For 0.1B scratch, they may not even be the right tool — the model fits on a CPU. Should we serve the 1.5B–9B base instead for W5?
- How do we measure "$ / 1M tokens" without picking a specific cloud? Pick a reference hardware (H100 PCIe, A100 80GB) and report both $/hr and throughput.