"""Tests for evaluation metrics: EER, MinDCF, bootstrap CI."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    bootstrap_ci,
    compute_all_metrics,
    compute_eer,
    compute_min_dcf,
)

# ─── compute_eer ─────────────────────────────────────────────────────────────


def test_eer_perfect_separation():
    """Perfect classifier should have EER near 0."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    y_score = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    eer = compute_eer(y_true, y_score)
    assert eer < 0.05


def test_eer_random_classifier():
    """Random classifier should have EER near 0.5."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 1000).astype(float)
    y_score = rng.rand(1000)
    eer = compute_eer(y_true, y_score)
    assert 0.35 < eer < 0.65


def test_eer_single_class_returns_nan():
    y_true = np.zeros(10)
    y_score = np.random.rand(10)
    eer = compute_eer(y_true, y_score)
    assert np.isnan(eer)


def test_eer_empty_returns_nan():
    eer = compute_eer(np.array([]), np.array([]))
    assert np.isnan(eer)


def test_eer_inverted_classifier():
    """Worst classifier (inverted scores) should still return valid EER near 1."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    y_score = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])  # inverted
    eer = compute_eer(y_true, y_score)
    assert np.isfinite(eer)


# ─── compute_min_dcf ─────────────────────────────────────────────────────────


def test_min_dcf_perfect():
    y_true = np.array([0, 0, 1, 1], dtype=float)
    y_score = np.array([0.1, 0.1, 0.9, 0.9])
    dcf = compute_min_dcf(y_true, y_score)
    assert dcf < 0.1


def test_min_dcf_random():
    rng = np.random.RandomState(1)
    y_true = rng.randint(0, 2, 500).astype(float)
    y_score = rng.rand(500)
    dcf = compute_min_dcf(y_true, y_score)
    assert 0.0 <= dcf <= 1.0


def test_min_dcf_single_class_nan():
    y_true = np.ones(10)
    y_score = np.random.rand(10)
    dcf = compute_min_dcf(y_true, y_score)
    assert np.isnan(dcf)


# ─── bootstrap_ci ────────────────────────────────────────────────────────────


def test_bootstrap_ci_contains_true_value():
    """95% CI should contain the true AUC most of the time."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 500).astype(float)
    y_score = y_true * 0.6 + rng.rand(500) * 0.4

    true_auc = roc_auc_score(y_true, y_score)
    _, lo, hi = bootstrap_ci(y_true, y_score, roc_auc_score, n_bootstrap=500, seed=42)

    assert lo <= true_auc <= hi, f"True AUC {true_auc:.3f} not in CI [{lo:.3f}, {hi:.3f}]"


def test_bootstrap_ci_lower_le_upper():
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 200).astype(float)
    y_score = rng.rand(200)
    _, lo, hi = bootstrap_ci(y_true, y_score, roc_auc_score, n_bootstrap=200)
    assert lo <= hi


# ─── compute_all_metrics ─────────────────────────────────────────────────────


def test_all_metrics_keys():
    y_true = np.array([0, 0, 1, 1], dtype=float)
    y_score = np.array([0.2, 0.3, 0.7, 0.8])
    result = compute_all_metrics(y_true, y_score, bootstrap=False)
    for key in ["EER", "AUC", "MinDCF", "AP", "F1", "Acc", "Prec", "Rec"]:
        assert key in result, f"Missing key: {key}"


def test_all_metrics_with_bootstrap():
    y_true = np.array([0] * 50 + [1] * 50, dtype=float)
    rng = np.random.RandomState(0)
    y_score = rng.rand(100)
    result = compute_all_metrics(y_true, y_score, bootstrap=True, n_bootstrap=100)
    assert "EER_CI_lo" in result
    assert "AUC_CI_lo" in result


def test_all_metrics_single_class_returns_nan():
    y_true = np.zeros(20)
    y_score = np.random.rand(20)
    result = compute_all_metrics(y_true, y_score, bootstrap=False)
    assert np.isnan(result["EER"])
    assert np.isnan(result["AUC"])
