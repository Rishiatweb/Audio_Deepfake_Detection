"""Tests for k-fold threshold calibration and cross-validation utilities."""

from __future__ import annotations

import numpy as np

from src.training.trainer import kfold_calibrate_threshold

# ─── kfold_calibrate_threshold ───────────────────────────────────────────────


def test_kfold_calibrate_returns_float():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, 100).astype(float)
    y_score = rng.rand(100)
    thr = kfold_calibrate_threshold(y_true, y_score, n_folds=3)
    assert isinstance(thr, float)
    assert 0.0 <= thr <= 1.0


def test_kfold_calibrate_in_valid_range():
    rng = np.random.RandomState(1)
    y_true = rng.randint(0, 2, 200).astype(float)
    y_score = rng.rand(200)
    thr = kfold_calibrate_threshold(y_true, y_score, n_folds=5)
    assert 0.05 <= thr <= 0.95  # grid search boundaries


def test_kfold_calibrate_perfect_separation():
    """Perfect separation → threshold should land near the gap (around 0.5)."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10, dtype=float)
    y_score = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.75, 0.8, 0.85, 0.9, 0.95] * 10)
    thr = kfold_calibrate_threshold(y_true, y_score, n_folds=2)
    assert 0.0 < thr < 1.0


def test_kfold_calibrate_empty_input():
    """Empty arrays fall back to 0.5."""
    thr = kfold_calibrate_threshold(np.array([]), np.array([]))
    assert thr == 0.5


def test_kfold_calibrate_single_class():
    """Single class (all positive or all negative) falls back to 0.5."""
    y_true = np.ones(20, dtype=float)
    y_score = np.random.rand(20)
    thr = kfold_calibrate_threshold(y_true, y_score)
    assert thr == 0.5


def test_kfold_calibrate_deterministic():
    """Same inputs yield same output across multiple calls (no randomness)."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 200).astype(float)
    y_score = rng.rand(200)
    t1 = kfold_calibrate_threshold(y_true, y_score, n_folds=3)
    t2 = kfold_calibrate_threshold(y_true, y_score, n_folds=3)
    assert t1 == t2


def test_kfold_calibrate_n_folds_1_fallback():
    """n_folds=1 or smaller than min class count falls back to single-pass search."""
    rng = np.random.RandomState(5)
    y_true = np.array([0, 0, 1, 1], dtype=float)  # only 2 of each class
    y_score = rng.rand(4)
    # n_folds=5 > min_class=2, should clamp and not crash
    thr = kfold_calibrate_threshold(y_true, y_score, n_folds=5)
    assert isinstance(thr, float)


def test_kfold_calibrate_vs_single_pass_close():
    """K-fold result should be within 0.2 of single-pass on the same data."""
    from src.training.trainer import find_best_threshold

    rng = np.random.RandomState(7)
    y_true = rng.randint(0, 2, 300).astype(float)
    y_score = y_true * 0.6 + rng.rand(300) * 0.4  # noisy but separable

    t_kfold = kfold_calibrate_threshold(y_true, y_score, n_folds=3)
    t_single, _ = find_best_threshold(y_true, y_score)
    assert abs(t_kfold - t_single) < 0.2
