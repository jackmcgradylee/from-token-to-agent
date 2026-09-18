"""W5 — Serving stack (vLLM / SGLang) + Load Generator.

Per COURSE §10, we measure TTFT / TPOT / P50 / P95 / cost-per-1M-tokens
on top of a mature serving engine, plus our own Load Generator that
sweeps concurrency / prompt length / generation length.
"""

__all__: list[str] = []