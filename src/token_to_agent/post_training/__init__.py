"""W8-W11 — Post-training at scale.

Per COURSE §13-§16, post-training experiments move to 1.5B-9B open
base models. The from_scratch/dpo and from_scratch/rlvr packages
own the *core loss / loop*; this package owns the *scale-run glue*
that calls TRL / LLaMA-Factory / slime / verl.
"""

__all__: list[str] = []