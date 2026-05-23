"""Tests for statistical tests and baseline model smoke checks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import load_config
from src.evaluation.metrics import compute_eer
from src.evaluation.statistical import delong_test, mcnemar_test, paired_bootstrap_test
from src.models.factory import get_model


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


@pytest.fixture
def dummy_mels(cfg):
    B = 2
    mels = []
    for mc in cfg.mel_configs:
        mels.append(torch.randn(B, 1, mc.n_mels, 251))
    return mels


# ─── McNemar's test ──────────────────────────────────────────────────────────


def test_mcnemar_identical_models():
    """Two identical classifiers: no significant difference."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=float)
    preds = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    result = mcnemar_test(y_true, preds, preds)
    assert not result["significant"]  # same predictions → not significant


def test_mcnemar_very_different_models():
    """One perfect, one random: should be significant."""
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 500).astype(float)
    preds_perfect = y_true.astype(int)
    preds_random = rng.randint(0, 2, 500)
    result = mcnemar_test(y_true, preds_perfect, preds_random)
    assert result["significant"]


def test_mcnemar_returns_keys():
    y_true = np.array([0, 1, 0, 1], dtype=float)
    preds_a = np.array([0, 1, 0, 0])
    preds_b = np.array([1, 1, 0, 1])
    result = mcnemar_test(y_true, preds_a, preds_b)
    for key in ["chi2", "p_value", "significant", "n01", "n10"]:
        assert key in result


# ─── DeLong's test ───────────────────────────────────────────────────────────


def test_delong_same_model_not_significant():
    rng = np.random.RandomState(1)
    y_true = rng.randint(0, 2, 200).astype(float)
    scores = rng.rand(200)
    result = delong_test(y_true, scores, scores)
    assert result["p_value"] >= 0.05


def test_delong_different_models_significant():
    rng = np.random.RandomState(2)
    y_true = rng.randint(0, 2, 500).astype(float)
    scores_good = y_true + rng.randn(500) * 0.1
    scores_bad = rng.rand(500)
    result = delong_test(y_true, scores_good, scores_bad)
    assert result["significant"]


def test_delong_returns_keys():
    rng = np.random.RandomState(3)
    y_true = rng.randint(0, 2, 100).astype(float)
    s_a = rng.rand(100)
    s_b = rng.rand(100)
    result = delong_test(y_true, s_a, s_b)
    for key in ["auc_a", "auc_b", "z_stat", "p_value", "significant"]:
        assert key in result


# ─── Paired bootstrap test ───────────────────────────────────────────────────


def test_paired_bootstrap_same_model():
    rng = np.random.RandomState(4)
    y_true = rng.randint(0, 2, 200).astype(float)
    scores = rng.rand(200)
    result = paired_bootstrap_test(y_true, scores, scores, compute_eer, n_bootstrap=200)
    assert abs(result["diff"]) < 1e-8
    assert result["p_value"] >= 0.1  # same model → not significant


# ─── Baseline model smoke tests ──────────────────────────────────────────────


@pytest.mark.parametrize("model_name", ["aasist", "lcnn"])
def test_baseline_forward_no_error(cfg, dummy_mels, model_name):
    model = get_model(model_name, cfg)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(dummy_mels)
    assert logits.shape == (dummy_mels[0].shape[0],)
    assert torch.isfinite(logits).all()


def test_rawnet2_forward_no_error(cfg):
    model = get_model("rawnet2", cfg)
    model.eval()
    B = 2
    wav = torch.randn(B, 1, cfg.audio.num_samples)
    dummy = [wav] + [torch.randn(B, 1, 80, 251)] * (len(cfg.mel_configs) - 1)
    with torch.no_grad():
        logits, _, _ = model(dummy)
    assert logits.shape == (B,)
    assert torch.isfinite(logits).all()


def test_all_baselines_output_same_batch_size(cfg, dummy_mels):
    B = dummy_mels[0].shape[0]
    for name in ["condetection", "aasist", "lcnn"]:
        model = get_model(name, cfg)
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(dummy_mels)
        assert logits.shape[0] == B, f"{name} batch size mismatch"
