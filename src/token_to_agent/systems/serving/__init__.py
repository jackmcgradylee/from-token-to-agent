"""systems.serving — runtime / scheduling layer for inference serving.

- load_generator: synthetic request arrival patterns.
- metrics_collector: per-request + aggregate latency stats.
- simulated_engine: CPU-side mock of a vLLM-style continuous-batching
  scheduler. Validates scheduling logic, not real forward passes.

Each module is independently importable; the experiment runner glues
them together.
"""

from .metrics_collector import (
    AggregateStats,
    MetricsCollector,
    RequestRecord,
    write_summary_csv,
    write_summary_md,
)
from .load_generator import (
    Request,
    generate_schedule,
    schedule_summary,
)
from .simulated_engine import (
    EngineConfig,
    SimulatedEngine,
)

__all__ = [
    "AggregateStats",
    "MetricsCollector",
    "RequestRecord",
    "write_summary_csv",
    "write_summary_md",
    "Request",
    "generate_schedule",
    "schedule_summary",
    "EngineConfig",
    "SimulatedEngine",
]
