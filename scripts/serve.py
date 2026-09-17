"""CLI entry: start a local OpenAI-compatible serving endpoint for the model.

This is a W5 stub. The actual serving is expected to integrate vLLM or SGLang;
for the 0.1B scratch baseline we can wrap a simple FastAPI server around the
in-process model. For 1.5B+ base, run vLLM directly (out of scope for this
script).

Usage (placeholder):
    PYTHONPATH=src python scripts/serve.py --ckpt ... --tokenizer ... --port 8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
for p in (str(_repo_root), str(_repo_root / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    print("[serve] W5 deliverable not yet implemented.")
    print("        For 1.5B+ models, prefer: vllm serve <model-id> --port 8080")
    print("        For 0.1B scratch baseline, the in-process /generate script suffices for W3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())