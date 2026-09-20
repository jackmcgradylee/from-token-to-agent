"""W3 scaling-pilot: 20M-class model (~33.4M actual params).

This is the biggest CPU-tolerable point in the W3 sweep. Expect ~10 min
wall time on the dev host.

Run:
    PYTHONPATH=. .venv/bin/python experiments/w03/exp-006-scaling-pilot-20m/runner.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _common as w3_common  # type: ignore  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _REPO_ROOT / "configs/hw1/scaling-20m.yaml"


def main() -> int:
    result = w3_common.run_scaling_pilot(
        config_path=CONFIG_PATH,
        output_dir=OUTPUT_DIR,
    )
    payload = asdict(result)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[exp-006] done: {payload['actual_params']:,d} params, "
          f"val_loss={payload['final_val_loss']:.4f}, "
          f"wall={payload['wall_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())