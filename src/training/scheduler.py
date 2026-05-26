"""Learning rate scheduling utilities."""

from __future__ import annotations

import math

import torch


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup."""

    def lr_fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
