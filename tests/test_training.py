"""Smoke tests for training loop and evaluation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.config import load_config
from src.models.factory import get_model
from src.training.losses import FocalBCEWithLogits
from src.training.scheduler import get_cosine_schedule_with_warmup
from src.training.trainer import evaluate, find_best_threshold, train_one_epoch


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_loader(cfg):
    """Minimal DataLoader with 4 batches of random waveforms."""

    n_samples = 32

    # Build a tiny DataFrame with zeros (no real files needed for smoke test)
    class _TinyDataset(torch.utils.data.Dataset):
        def __init__(self, n, num_samples):
            self.n = n
            self.num_samples = num_samples

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            wav = torch.randn(self.num_samples) * 0.1
            lbl = torch.tensor(float(idx % 2), dtype=torch.float32)
            return wav, lbl

    ds = _TinyDataset(n_samples, cfg.audio.num_samples)
    return DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)


# ─── find_best_threshold ─────────────────────────────────────────────────────


def test_find_best_threshold_valid():
    y_true = np.array([0, 0, 1, 1, 0, 1], dtype=float)
    y_score = np.array([0.2, 0.3, 0.7, 0.8, 0.4, 0.6])
    t, obj = find_best_threshold(y_true, y_score)
    assert 0.0 < t < 1.0
    assert obj >= 0.0


def test_find_best_threshold_single_class():
    y_true = np.zeros(10)
    y_score = np.random.rand(10)
    t, obj = find_best_threshold(y_true, y_score)
    assert t == 0.5
    assert obj == 0.0


# ─── Cosine scheduler ────────────────────────────────────────────────────────


def test_cosine_scheduler_warmup():
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    get_cosine_schedule_with_warmup(opt, warmup_steps=10, total_steps=100)
    # At step 0 lr should be ~0
    assert opt.param_groups[0]["lr"] < 1e-3


def test_cosine_scheduler_reaches_peak():
    model = torch.nn.Linear(4, 1)
    lr0 = 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr0)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps=10, total_steps=100)
    for _ in range(10):
        sched.step()
    # After warmup lr should equal lr0 (multiplier ~1.0)
    assert abs(opt.param_groups[0]["lr"] - lr0) < 1e-5


# ─── Training smoke test ─────────────────────────────────────────────────────


def test_train_one_epoch_loss_decreases(cfg, device, tiny_loader):
    """Loss should not be NaN and should be finite after one epoch."""
    model = get_model("condetection", cfg).to(device)
    criterion = FocalBCEWithLogits(gamma=1.5, alpha=0.6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 5, 50)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    loss = train_one_epoch(
        model,
        tiny_loader,
        optimizer,
        scheduler,
        scaler,
        criterion,
        device,
        cfg,
        epoch=1,
    )
    assert math.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss >= 0


def test_evaluate_returns_metrics(cfg, device, tiny_loader):
    model = get_model("condetection", cfg).to(device)
    criterion = FocalBCEWithLogits(gamma=1.5, alpha=0.6)
    result = evaluate(model, tiny_loader, criterion, device, cfg, threshold=0.5)
    for key in ["loss", "AUC", "EER", "F1"]:
        assert key in result
    assert math.isfinite(result["loss"])


def test_evaluate_no_nan_loss(cfg, device, tiny_loader):
    model = get_model("condetection", cfg).to(device)
    criterion = FocalBCEWithLogits()
    result = evaluate(model, tiny_loader, criterion, device, cfg)
    assert math.isfinite(result["loss"])
