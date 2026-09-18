"""W9 — DPO core loss (hand-written, teaching version).

Full-scale training wraps this with TRL / LLaMA-Factory (see
src/token_to_agent/post_training/preference/). Per COURSE §3, we
must own the chosen/rejected log-prob and reference-policy ratio math
even if we use a framework to run thousands of steps.
"""

__all__: list[str] = []