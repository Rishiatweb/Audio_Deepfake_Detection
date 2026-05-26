"""Loss functions for ConDetection-DANN training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config


class FocalBCEWithLogits(nn.Module):
    """Focal Binary Cross-Entropy for class imbalance.

    Reference: Lin et al. (2017) — Focal Loss for Dense Object Detection.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.45, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("pos_weight_buf", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight_buf)
        p = torch.sigmoid(logits)
        pt = targets * p + (1.0 - targets) * (1.0 - p)
        focal = (1.0 - pt).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
            focal = focal * alpha_t
        return (focal * bce).mean()


def label_smooth(labels: torch.Tensor, s: float = 0.03) -> torch.Tensor:
    """Soft labels: push targets away from hard 0/1."""
    return labels * (1 - s) + 0.5 * s


def build_dann_domain_labels(batch_size: int, is_source: bool, device: torch.device) -> torch.Tensor:
    """Soft domain labels: 0.05=source(FoR), 0.95=target(ITW)."""
    lbl = 0.05 if is_source else 0.95
    return torch.full((batch_size,), lbl, dtype=torch.float32, device=device)


def dann_lambda_schedule(
    epoch: int,
    total_epochs: int,
    warmup: int = 4,
    max_lambda: float = 0.3,
) -> float:
    """Anneal DANN gradient reversal strength (Ganin et al. 2016).

    Zero during warmup, then sigmoid ramp to max_lambda.
    """
    if epoch <= warmup:
        return 0.0
    p = (epoch - warmup) / max(1, total_epochs - warmup)
    return float(max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))


def build_criterion(cfg: Config, device: torch.device, pos_weight_val: float | None = None) -> nn.Module:
    """Build main classification criterion from config."""
    pos_w = None
    if pos_weight_val is not None:
        pos_w = torch.tensor([max(pos_weight_val, 1e-3)], device=device)

    if cfg.training.use_focal_loss:
        return FocalBCEWithLogits(
            gamma=cfg.training.focal_gamma,
            alpha=cfg.training.focal_alpha,
            pos_weight=pos_w,
        )
    return nn.BCEWithLogitsLoss(pos_weight=pos_w)
