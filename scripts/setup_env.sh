#!/bin/bash
# Set up the local Python environment for this repo.
# Idempotent: safe to re-run.
#
# Usage: bash scripts/setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

echo "[setup] repo: $REPO_ROOT"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [[ ! -d ".venv" ]]; then
  echo "[setup] creating venv with python $PYTHON_VERSION"
  uv venv .venv --python "$PYTHON_VERSION"
else
  echo "[setup] venv exists"
fi

.venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 || true

# CPU-only torch + minimal deps. --only-binary=:all: forces wheels
# (avoids source-build / cmake failures on the dev host / constrained Linux).
.venv/bin/python -m pip install --upgrade --only-binary=:all: pip setuptools wheel
.venv/bin/python -m pip install --only-binary=:all: torch pyyaml pandas matplotlib numpy

# Reminder for users: PYTHONPATH=. is required for `from src.token_to_agent.* import ...`
echo
echo "[setup] done."
echo "        Activate with:    source .venv/bin/activate"
echo "        Run scripts with: PYTHONPATH=. .venv/bin/python scripts/<name>.py"
echo "        Run tests with:   PYTHONPATH=. .venv/bin/python tests/<...>/test_<...>.py"