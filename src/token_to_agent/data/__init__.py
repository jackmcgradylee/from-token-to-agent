"""W6-W7 — Raw data pipeline + Synthetic Data Factory.

Per COURSE §11: Raw Dump -> Parse -> Normalize -> Language Filter
-> Quality Filter -> Exact Dedup -> Near Dedup -> Contamination
Check -> Mixture -> Tokenizer -> Training Shards.

Per COURSE §12: Task Generator -> Candidate Task -> Solver ->
Verifier/Judge -> Filter -> Training Data.
"""

__all__: list[str] = []