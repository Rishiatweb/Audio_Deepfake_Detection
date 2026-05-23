"""Statistical significance tests for model comparison."""

from __future__ import annotations

import numpy as np


def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    """McNemar's test: are two classifiers significantly different?

    Tests whether model A and model B make systematically different errors
    at the sample level.

    Args:
        y_true: ground-truth binary labels
        preds_a: binary predictions from model A
        preds_b: binary predictions from model B

    Returns:
        dict with chi2, p_value, n01 (A wrong, B right), n10 (A right, B wrong)
    """
    from statsmodels.stats.contingency_tables import mcnemar

    y_true = np.asarray(y_true).reshape(-1)
    preds_a = np.asarray(preds_a).reshape(-1)
    preds_b = np.asarray(preds_b).reshape(-1)

    correct_a = (preds_a == y_true).astype(int)
    correct_b = (preds_b == y_true).astype(int)

    # Contingency table: [both correct, A wrong B right; A right B wrong, both wrong]
    n00 = int(((correct_a == 0) & (correct_b == 0)).sum())
    n01 = int(((correct_a == 0) & (correct_b == 1)).sum())  # A wrong, B right
    n10 = int(((correct_a == 1) & (correct_b == 0)).sum())  # A right, B wrong
    n11 = int(((correct_a == 1) & (correct_b == 1)).sum())

    table = np.array([[n11, n10], [n01, n00]])
    result = mcnemar(table, exact=False, correction=True)

    return {
        "chi2": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < 0.05),
        "n01": n01,
        "n10": n10,
        "table": table,
    }


def delong_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """DeLong's test for comparing two ROC AUC values.

    Reference: DeLong et al. (1988) — Comparing the Areas under Two or More
    Correlated Receiver Operating Characteristic Curves.

    Args:
        y_true: ground-truth binary labels
        scores_a: continuous scores from model A
        scores_b: continuous scores from model B

    Returns:
        dict with auc_a, auc_b, z_stat, p_value, significant
    """
    from scipy import stats
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=float)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return {"auc_a": np.nan, "auc_b": np.nan, "z_stat": np.nan, "p_value": np.nan, "significant": False}

    def structural_components(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """V10 and V01 structural components for DeLong variance."""
        m = len(pos_idx)
        n = len(neg_idx)
        v10 = np.zeros(m)
        v01 = np.zeros(n)
        for i, pi in enumerate(pos_idx):
            v10[i] = np.mean(scores[pi] > scores[neg_idx]) + 0.5 * np.mean(scores[pi] == scores[neg_idx])
        for j, ni in enumerate(neg_idx):
            v01[j] = np.mean(scores[pos_idx] > scores[ni]) + 0.5 * np.mean(scores[pos_idx] == scores[ni])
        return v10, v01

    v10_a, v01_a = structural_components(scores_a)
    v10_b, v01_b = structural_components(scores_b)

    auc_a = roc_auc_score(y_true, scores_a)
    auc_b = roc_auc_score(y_true, scores_b)

    m = len(pos_idx)
    n = len(neg_idx)

    # Covariance matrix of [AUC_a, AUC_b]
    s_aa = np.var(v10_a) / m + np.var(v01_a) / n
    s_bb = np.var(v10_b) / m + np.var(v01_b) / n
    s_ab = np.cov(v10_a, v10_b)[0, 1] / m + np.cov(v01_a, v01_b)[0, 1] / n

    var_diff = s_aa + s_bb - 2 * s_ab
    if var_diff <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "z_stat": 0.0, "p_value": 1.0, "significant": False}

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "z_stat": float(z),
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test: is metric(A) significantly different from metric(B)?

    Args:
        metric_fn: callable(y_true, y_score) → float (e.g., compute_eer)

    Returns:
        dict with metric_a, metric_b, p_value, significant
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    point_a = metric_fn(y_true, scores_a)
    point_b = metric_fn(y_true, scores_b)
    observed_diff = point_a - point_b

    # Permutation under null hypothesis: A and B are equivalent
    count_extreme = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if np.unique(yt).size < 2:
            continue
        try:
            boot_diff = metric_fn(yt, scores_a[idx]) - metric_fn(yt, scores_b[idx])
            if abs(boot_diff) >= abs(observed_diff):
                count_extreme += 1
        except Exception:
            continue

    p_value = count_extreme / n_bootstrap

    return {
        "metric_a": float(point_a),
        "metric_b": float(point_b),
        "diff": float(observed_diff),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
