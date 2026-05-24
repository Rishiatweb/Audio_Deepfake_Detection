"""Tests for loss functions."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.config import load_config
from src.training.losses import (
    FocalBCEWithLogits,
    build_criterion,
    build_dann_domain_labels,
    dann_lambda_schedule,
    label_smooth,
)


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


# ─── FocalBCEWithLogits ──────────────────────────────────────────────────────


def test_focal_bce_reduces_to_bce_when_gamma_zero():
    """With gamma=0, focal loss == standard BCE."""
    focal = FocalBCEWithLogits(gamma=0.0, alpha=None)
    std = nn.BCEWithLogitsLoss()
    logits = torch.randn(16)
    targets = torch.randint(0, 2, (16,)).float()
    assert torch.allclose(focal(logits, targets), std(logits, targets), atol=1e-5)


def test_focal_bce_gradient_exists():
    focal = FocalBCEWithLogits(gamma=1.5, alpha=0.6)
    logits = torch.randn(8, requires_grad=True)
    targets = torch.randint(0, 2, (8,)).float()
    loss = focal(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_bce_positive():
    focal = FocalBCEWithLogits(gamma=1.5, alpha=0.6)
    logits = torch.randn(8)
    targets = torch.randint(0, 2, (8,)).float()
    assert focal(logits, targets) >= 0


def test_focal_bce_with_pos_weight():
    pos_w = torch.tensor([2.0])
    focal = FocalBCEWithLogits(gamma=1.5, alpha=0.6, pos_weight=pos_w)
    logits = torch.randn(8)
    targets = torch.randint(0, 2, (8,)).float()
    loss = focal(logits, targets)
    assert torch.isfinite(loss)


# ─── label_smooth ────────────────────────────────────────────────────────────


def test_label_smooth_range():
    labels = torch.zeros(10)
    smoothed = label_smooth(labels, s=0.1)
    assert (smoothed >= 0).all() and (smoothed <= 1).all()


def test_label_smooth_hard_ones():
    labels = torch.ones(10)
    smoothed = label_smooth(labels, s=0.1)
    assert (smoothed < 1.0).all()
    assert (smoothed > 0.5).all()


# ─── build_dann_domain_labels ────────────────────────────────────────────────


def test_dann_labels_source():
    labels = build_dann_domain_labels(8, is_source=True, device=torch.device("cpu"))
    assert labels.shape == (8,)
    assert (labels < 0.5).all()  # soft 0 for source


def test_dann_labels_target():
    labels = build_dann_domain_labels(8, is_source=False, device=torch.device("cpu"))
    assert (labels > 0.5).all()  # soft 1 for target


# ─── dann_lambda_schedule ────────────────────────────────────────────────────


def test_dann_lambda_zero_during_warmup():
    for epoch in range(1, 5):
        lam = dann_lambda_schedule(epoch, total_epochs=20, warmup=4, max_lambda=0.3)
        assert lam == 0.0


def test_dann_lambda_increases_after_warmup():
    lam_early = dann_lambda_schedule(5, total_epochs=20, warmup=4, max_lambda=0.3)
    lam_late = dann_lambda_schedule(15, total_epochs=20, warmup=4, max_lambda=0.3)
    assert lam_late > lam_early


def test_dann_lambda_bounded_by_max():
    for epoch in range(1, 25):
        lam = dann_lambda_schedule(epoch, total_epochs=20, warmup=4, max_lambda=0.3)
        assert 0.0 <= lam <= 0.3 + 1e-6


# ─── build_criterion ─────────────────────────────────────────────────────────


def test_build_criterion_focal(cfg):
    device = torch.device("cpu")
    criterion = build_criterion(cfg, device, pos_weight_val=1.5)
    assert isinstance(criterion, FocalBCEWithLogits)


def test_build_criterion_standard_bce(cfg):
    cfg.training.use_focal_loss = False
    device = torch.device("cpu")
    criterion = build_criterion(cfg, device)
    assert isinstance(criterion, nn.BCEWithLogitsLoss)
