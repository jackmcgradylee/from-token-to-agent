"""Training loop and utilities."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .scheduler import cosine_lr


@dataclass
class TrainConfig:
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    batch_size: int = 8
    grad_accum: int = 4
    max_steps: int = 1000
    log_every: int = 20
    eval_every: int = 200
    ckpt_dir: str = "checkpoints/run"
    log_csv: str = "loss_curve.csv"
    precision: str = "fp32"  # "fp32" | "bf16" | "fp16"
    seed: int = 42


class TokenDataset(Dataset):
    """In-memory dataset of token IDs, sliced into fixed-length windows.

    For HW1: load a tokenized text file into memory once and slice into
    (seq_len+1) sequences. Targets = inputs shifted by 1 (next-token LM).
    """

    def __init__(self, ids: list[int], seq_len: int):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(len(self.ids) - self.seq_len - 1, 0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def train(
    model: nn.Module,
    train_ids: list[int],
    val_ids: list[int],
    cfg: TrainConfig,
    device: str = "cpu",
    log_fn=None,
) -> dict:
    """Run the training loop.

    Returns: dict with final train_loss, val_loss, tokens_per_sec, peak_mem_mb.
    """
    log_fn = log_fn or (lambda *a, **kw: None)
    torch.manual_seed(cfg.seed)
    model.to(device)
    model.train()

    train_ds = TokenDataset(train_ids, seq_len=model.cfg.max_seq_len)
    val_ds = TokenDataset(val_ids, seq_len=model.cfg.max_seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    autocast_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[cfg.precision]

    step = 0
    losses = []
    tokens_per_sec_window = []
    t_start = time.time()
    peak_mem = 0

    log_path = os.path.join(cfg.ckpt_dir, cfg.log_csv)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("step,train_loss,val_loss,lr,tokens_per_sec,wall_s\n")

    while step < cfg.max_steps:
        for x, y in train_dl:
            if step >= cfg.max_steps:
                break
            x = x.to(device)
            y = y.to(device)

            lr = cosine_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr_ratio)
            for pg in optim.param_groups:
                pg["lr"] = lr

            optim.zero_grad(set_to_none=True)

            t0 = time.time()
            with torch.amp.autocast("cpu", enabled=False):
                _, loss = model(x, targets=y)

            loss = loss / cfg.grad_accum
            loss.backward()

            do_step = (step + 1) % cfg.grad_accum == 0
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

            dt = time.time() - t0
            tokens = x.numel()
            tokens_per_sec = tokens / max(dt, 1e-6)
            tokens_per_sec_window.append(tokens_per_sec)
            losses.append(loss.item() * cfg.grad_accum)

            if device == "cuda":
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e6)

            if step % cfg.log_every == 0:
                avg_loss = sum(losses[-cfg.log_every:]) / min(len(losses), cfg.log_every)
                avg_tps = sum(tokens_per_sec_window[-cfg.log_every:]) / max(
                    len(tokens_per_sec_window), 1
                )
                wall_s = time.time() - t_start
                with open(log_path, "a") as f:
                    f.write(f"{step},{avg_loss:.4f},,{lr:.2e},{avg_tps:.1f},{wall_s:.1f}\n")
                log_fn(
                    f"step={step:5d} loss={avg_loss:.4f} lr={lr:.2e} "
                    f"tps={avg_tps:.1f} wall={wall_s:.0f}s"
                )

            # Eval
            if step > 0 and step % cfg.eval_every == 0:
                val_loss = evaluate(model, val_ds, cfg.batch_size, device)
                wall_s = time.time() - t_start
                with open(log_path, "a") as f:
                    f.write(f"{step},,{val_loss:.4f},,,{wall_s:.1f}\n")
                log_fn(f"  eval @ step={step}: val_loss={val_loss:.4f}")

            step += 1

    # Final eval
    final_val = evaluate(model, val_ds, cfg.batch_size, device)
    with open(log_path, "a") as f:
        f.write(f"{step},,{final_val:.4f},,,{time.time() - t_start:.1f}\n")

    # Save final checkpoint
    ckpt_path = os.path.join(cfg.ckpt_dir, "model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "cfg": model.cfg.__dict__,
        "step": step,
    }, ckpt_path)
    log_fn(f"saved checkpoint to {ckpt_path}")

    return {
        "final_train_loss": sum(losses[-cfg.log_every:]) / min(len(losses), cfg.log_every),
        "final_val_loss": final_val,
        "tokens_per_sec_avg": sum(tokens_per_sec_window) / max(len(tokens_per_sec_window), 1),
        "peak_mem_mb": peak_mem,
        "total_wall_s": time.time() - t_start,
        "steps": step,
    }


@torch.no_grad()
def evaluate(model: nn.Module, ds: TokenDataset, batch_size: int, device: str) -> float:
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    total_loss = 0.0
    total_count = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
    model.train()
    return total_loss / max(total_count, 1)