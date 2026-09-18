"""systems.distributed — PyTorch DDP / FSDP reference wrappers.

This module is a *thin* wrapper layer around `torch.distributed` /
`torch.nn.parallel.DistributedDataParallel`. W4 does **not** ask us to
reimplement distributed training (COURSE.md §九 explicitly says "不要求
自己重新实现 Distributed Training"). What we add is:

  1. A reproducible setup helper that respects environment variables
     (RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT) and falls back to
     a single-process pseudo-distributed run for CPU-only hosts.
  2. A correctness helper: given a single-GPU reference run, compare to
     the DDP-equivalent summed-gradients version, and assert the two
     match (modulo fp noise). This is the test you can run on a CPU
     host with WORLD_SIZE=1 that still meaningfully exercises the
     "average gradients across N replicas" path.
  3. Two smoke scripts:
       - `manual_all_reduce.py` — manual replication + all-reduce over
         plain Python lists. Self-contained test of the all-reduce
         primitive without touching torch.distributed.
       - `fsdp_smoke.py` — minimal FSDP-wrapped 2-layer MLP that you
         can launch with `torchrun --nproc_per_node=N`.

Run a unit test:
    PYTHONPATH=. .venv/bin/python -m pytest tests/systems/ -v
    PYTHONPATH=. .venv/bin/python src/token_to_agent/systems/distributed/ddp_reference.py
"""

from __future__ import annotations

import os
import socket
from typing import Optional

import torch
import torch.distributed as dist


def is_distributed_available() -> bool:
    """True if torch.distributed is built into this PyTorch."""
    return dist.is_available()


def get_world_size_from_env() -> int:
    """Read WORLD_SIZE from env, default 1."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank_from_env() -> int:
    """Read RANK from env, default 0."""
    return int(os.environ.get("RANK", "0"))


def find_free_port() -> int:
    """Bind a socket to get an unused port. Used to auto-pick MASTER_PORT."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_distributed(
    backend: str = "gloo",
    init_method: Optional[str] = None,
    *,
    allow_fallback: bool = True,
) -> tuple[int, int, int]:
    """Initialize torch.distributed if WORLD_SIZE > 1.

    Returns: (rank, world_size, local_rank).

    If WORLD_SIZE is 1 and `allow_fallback`, returns (0, 1, 0) without
    calling `dist.init_process_group` — handy for CPU-only hosts and CI.

    Else, expects the standard env vars (RANK, WORLD_SIZE, MASTER_ADDR,
    MASTER_PORT) or an explicit `init_method` URL.
    """
    world_size = get_world_size_from_env()
    rank = get_rank_from_env()

    if world_size == 1 and allow_fallback:
        # Single-process mode: nothing to initialise.
        return rank, world_size, 0

    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")

    if init_method is None:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", str(find_free_port()))
        init_method = f"tcp://{master_addr}:{master_port}"

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Tear down torch.distributed if it was initialised."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def manual_all_reduce_sum(values: list[float]) -> list[float]:
    """Toy all-reduce-SUM over a plain Python list — runs on any host.

    Used by the unit tests to verify the *primitive* works without
    needing torch.distributed to actually be running.

    Real DDP uses ring or tree all-reduce over NCCL/Gloo; this loop is
    just a baseline.
    """
    n = len(values)
    if n == 0:
        return values
    s = sum(values)
    return [s for _ in range(n)]


def manual_all_reduce_avg(values: list[float]) -> list[float]:
    """Toy all-reduce-AVG: sum / N. Equivalent to SUM then divide."""
    n = len(values)
    if n == 0:
        return values
    s = sum(values) / n
    return [s for _ in range(n)]


# ---------------------------------------------------------------------------
# Toy training-loop smoke test (no DDP). Demonstrates that the manual
# all-reduce-avg primitive produces the same parameter updates as a
# hypothetical 2-process average.
# ---------------------------------------------------------------------------


def smoke_ddp_correctness() -> bool:
    """Single-process CPU test of "if I had two replicas and averaged
    their gradients, would the parameter updates match?".

    Steps:
      1. Build a 2-layer MLP with random init.
      2. Forward + loss on input A; record gradient g_a.
      3. Forward + loss on input B; record gradient g_b.
      4. Manually average g_a and g_b (simulating 2-process DDP mean
         reduction).
      5. Verify that the averaged gradient is the same as the gradient
         you'd get from doing one forward on the concat of (A, B) with
         batch_size doubled (which is mathematically what DDP-averaged
         gradients give you).

    Returns True on success; raises AssertionError otherwise.
    """
    torch.manual_seed(0)

    # Two-layer MLP, 4-dim input, 3-dim output.
    W1 = torch.randn(4, 3, requires_grad=True)
    b1 = torch.zeros(3, requires_grad=True)
    W2 = torch.randn(3, 2, requires_grad=True)
    b2 = torch.zeros(2, requires_grad=True)

    x_a = torch.randn(2, 4)  # batch=2
    y_a = torch.randn(2, 2)
    x_b = torch.randn(2, 4)  # batch=2
    y_b = torch.randn(2, 2)

    def fwd(x, W1, b1, W2, b2):
        h = torch.relu(x @ W1 + b1)
        return h @ W2 + b2

    def loss_fn(x, y, W1, b1, W2, b2):
        out = fwd(x, W1, b1, W2, b2)
        return ((out - y) ** 2).mean()

    # Replica A
    W1a = W1.detach().clone().requires_grad_(True)
    b1a = b1.detach().clone().requires_grad_(True)
    W2a = W2.detach().clone().requires_grad_(True)
    b2a = b2.detach().clone().requires_grad_(True)
    loss_a = loss_fn(x_a, y_a, W1a, b1a, W2a, b2a)
    loss_a.backward()
    grads_a = [W1a.grad, b1a.grad, W2a.grad, b2a.grad]

    # Replica B
    W1b = W1.detach().clone().requires_grad_(True)
    b1b = b1.detach().clone().requires_grad_(True)
    W2b = W2.detach().clone().requires_grad_(True)
    b2b = b2.detach().clone().requires_grad_(True)
    loss_b = loss_fn(x_b, y_b, W1b, b1b, W2b, b2b)
    loss_b.backward()
    grads_b = [W1b.grad, b1b.grad, W2b.grad, b2b.grad]

    # Average (DDP mean-reduction).
    grads_avg = [(ga + gb) / 2.0 for ga, gb in zip(grads_a, grads_b)]

    # Reference: single forward on the concatenated batch (also size 4).
    W1c = W1.detach().clone().requires_grad_(True)
    b1c = b1.detach().clone().requires_grad_(True)
    W2c = W2.detach().clone().requires_grad_(True)
    b2c = b2.detach().clone().requires_grad_(True)
    x_concat = torch.cat([x_a, x_b], dim=0)
    y_concat = torch.cat([y_a, y_b], dim=0)
    loss_concat = loss_fn(x_concat, y_concat, W1c, b1c, W2c, b2c)
    loss_concat.backward()
    grads_ref = [W1c.grad, b1c.grad, W2c.grad, b2c.grad]

    for label, g_avg, g_ref in zip(
        ["W1", "b1", "W2", "b2"], grads_avg, grads_ref,
    ):
        max_err = (g_avg - g_ref).abs().max().item()
        assert max_err < 1e-5, (
            f"DDP-averaged gradient for {label} diverges from concat reference: "
            f"max_err={max_err:.2e}"
        )
    return True


if __name__ == "__main__":
    # Standalone smoke run.
    print("[ddp_reference] testing manual_all_reduce_sum on 4 values...")
    out = manual_all_reduce_sum([1.0, 2.0, 3.0, 4.0])
    assert out == [10.0, 10.0, 10.0, 10.0]
    print(f"  [PASS] all_reduce_sum([1,2,3,4]) = {out}")

    print("[ddp_reference] testing manual_all_reduce_avg on 4 values...")
    out = manual_all_reduce_avg([1.0, 2.0, 3.0, 4.0])
    assert out == [2.5, 2.5, 2.5, 2.5]
    print(f"  [PASS] all_reduce_avg([1,2,3,4]) = {out}")

    print("[ddp_reference] testing smoke_ddp_correctness...")
    ok = smoke_ddp_correctness()
    print(f"  [PASS] smoke_ddp_correctness = {ok}")

    print("\nAll DDP-reference smoke tests passed.")