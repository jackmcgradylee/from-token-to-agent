# W4 / Triton vs PyTorch vs Reference benchmark (dev CPU, fp32)

See `benchmark-cpu.csv` for the full table; `metadata.json` for the structured summary.

## Bottom line

PyTorch SDPA (`F.scaled_dot_product_attention`) beats the pure-PyTorch reference by **1.46×–4.14×** on dev CPU fp32 across the B × T sweep. Numerical error vs the reference is at the fp32 noise floor (max abs error ≤ 1.49e-7).

Triton kernel (`src/token_to_agent/kernels/triton_attention.py`) is **skipped on the dev host** — requires CUDA. Re-run on a GPU box to fill in the Triton row.