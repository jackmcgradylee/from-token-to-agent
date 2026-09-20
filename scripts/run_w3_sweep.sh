#!/usr/bin/env bash
# Run the W3 scaling pilot sweep end-to-end (5 pilots + power-law fit).
#
# What this does:
#   0. (one-time) regenerate the 1 MB toy corpus if data/raw/sample.txt is missing
#   1. Train 5 iso-data scaling pilots: 1m / 2_5m / 5m / 10m / 20m
#   2. Fit L(N) = L_inf + a * N^(-alpha) on {1m, 2_5m, 5m, 10m}
#   3. Predict the held-out 20m point and report relative error
#   4. Render scaling_curve.png + fit.md + fit.json under
#      experiments/w03/exp-007-scaling-law-fit/results/
#
# Wall time on the dev host CPU: ~40 minutes for the sweep + ~1 s for the fit.
#
# Usage:
#   bash scripts/run_w3_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="."

echo "[w3-sweep] repo: $REPO_ROOT"
echo "[w3-sweep] Python: $(.venv/bin/python --version 2>&1 || echo 'no .venv found')"

# 0. corpus
if [[ ! -f data/raw/sample.txt ]]; then
    echo "[w3-sweep] generating data/raw/sample.txt (1 MiB TinyStories-style, seed=42)"
    .venv/bin/python scripts/make_sample_corpus.py
fi

# 1. 5 pilots — sequential on CPU
for exp in \
    experiments/w03/exp-002-scaling-pilot-1m \
    experiments/w03/exp-003-scaling-pilot-2_5m \
    experiments/w03/exp-004-scaling-pilot-5m \
    experiments/w03/exp-005-scaling-pilot-10m \
    experiments/w03/exp-006-scaling-pilot-20m; do
    echo "[w3-sweep] === $exp ==="
    .venv/bin/python "$exp/runner.py"
done

# 2. fit + hold-out
echo "[w3-sweep] === exp-007-scaling-law-fit ==="
.venv/bin/python experiments/w03/exp-007-scaling-law-fit/runner.py

echo "[w3-sweep] done. See:"
echo "  - experiments/w03/exp-007-scaling-law-fit/results/fit.md"
echo "  - experiments/w03/exp-007-scaling-law-fit/results/scaling_curve.png"