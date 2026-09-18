"""W4 — Distributed training wrappers (PyTorch DDP / FSDP / Megatron).

Per COURSE §3, this is framework territory: we don't reimplement
collective primitives, but we do own the launch & config glue and
benchmark the speedup honestly.
"""

__all__: list[str] = []