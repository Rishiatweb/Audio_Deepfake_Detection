"""Evaluation metrics: EER, MinDCF, bootstrap CIs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Equal Error Rate: threshold where FAR == FRR.

    Returns NaN if labels are single-class or inputs are empty.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    n = min(len(y_true), len(y_score))
    y_true = y_true[:n]
    y_score = y_score[:n]
    finite = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return np.nan

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    d = np.abs(fnr - fpr)
    finite_d = np.isfinite(d)
    if not finite_d.any():
        return np.nan
    idx = np.argmin(d[finite_d])
    return float((fpr[finite_d][idx] + fnr[finite_d][idx]) / 2)


def compute_min_dcf(
    y_true: np.ndarray,
    y_score: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Minimum Detection Cost Function (MinDCF).

    Standard metric in ASVspoof challenge evaluation.
    DCF(t) = C_miss * P_miss(t) * P_target + C_fa * P_fa(t) * (1 - P_target)
    MinDCF = min over all thresholds t.

    Returns NaN if labels are single-class or inputs are empty.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    finite = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[finite], y_score[finite]

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return np.nan

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    return float(np.nanmin(dcf))


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for any scalar metric.

    Args:
        y_true: ground-truth binary labels
        y_score: continuous prediction scores
        metric_fn: callable(y_true, y_score) → float
        n_bootstrap: number of resamples
        confidence: CI level

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    point = metric_fn(y_true, y_score)

    boot_vals = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if np.unique(yt).size < 2:
            continue
        try:
            v = metric_fn(yt, ys)
            if np.isfinite(v):
                boot_vals.append(v)
        except Exception:
            continue

    if len(boot_vals) < 10:
        return float(point), float("nan"), float("nan")

    alpha = 1 - confidence
    lo = float(np.percentile(boot_vals, 100 * alpha / 2))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    bootstrap: bool = True,
    n_bootstrap: int = 2000,
) -> dict:
    """Compute full metric suite: EER, AUC, MinDCF, F1, with bootstrap CIs.

    Args:
        y_true: binary ground-truth
        y_score: continuous scores
        threshold: decision threshold for precision/recall/F1
        bootstrap: compute 95% CIs via bootstrap
        n_bootstrap: bootstrap resamples

    Returns:
        dict with all metrics and optional CI bounds
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    finite = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[finite], y_score[finite]

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return {
            "EER": np.nan,
            "AUC": np.nan,
            "MinDCF": np.nan,
            "AP": np.nan,
            "F1": 0.0,
            "Acc": 0.0,
            "Prec": 0.0,
            "Rec": 0.0,
        }

    y_pred = (y_score >= threshold).astype(int)

    result: dict = {
        "EER": compute_eer(y_true, y_score),
        "AUC": roc_auc_score(y_true, y_score),
        "MinDCF": compute_min_dcf(y_true, y_score),
        "AP": average_precision_score(y_true, y_score),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Acc": accuracy_score(y_true, y_pred),
        "Prec": precision_score(y_true, y_pred, zero_division=0),
        "Rec": recall_score(y_true, y_pred, zero_division=0),
    }

    if bootstrap:
        for metric_name, fn in [
            ("EER", compute_eer),
            ("AUC", roc_auc_score),
            ("MinDCF", compute_min_dcf),
        ]:
            _, lo, hi = bootstrap_ci(y_true, y_score, fn, n_bootstrap)
            result[f"{metric_name}_CI_lo"] = lo
            result[f"{metric_name}_CI_hi"] = hi

    return result
