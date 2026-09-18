"""W10 — RLVR teaching loop (hand-written K-rollouts + group advantage).

Full-scale training runs through slime / verl / TRL
(src/token_to_agent/post_training/rl/). Per COURSE §3, we must
own the prompt -> K rollouts -> verifier -> reward -> group advantage
-> policy loss -> update cycle at least once at small scale.
"""

__all__: list[str] = []