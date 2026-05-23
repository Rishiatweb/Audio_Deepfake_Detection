"""Publication figures: training curves, ROC, confusion matrices, comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

matplotlib.use("Agg")

STYLE = {
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
}


def _save(fig: plt.Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_training_curves(history: list[dict], out_path: str) -> None:
    """Plot loss, EER, AUC, F1, DANN lambda over training epochs."""
    with plt.rc_context(STYLE):
        df = pd.DataFrame(history)
        metrics = [
            ("train_loss", "val_loss", "Loss"),
            ("EER", None, "EER (↓)"),
            ("AUC", None, "AUC (↑)"),
            ("F1", None, "F1 (↑)"),
            ("dann_lambda", None, "DANN λ"),
        ]
        valid = [(a, b, t) for a, b, t in metrics if a in df.columns]
        n = len(valid)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (col_a, col_b, title) in zip(axes, valid):
            ax.plot(df["epoch"], df[col_a], label=col_a, color="steelblue")
            if col_b and col_b in df.columns:
                ax.plot(df["epoch"], df[col_b], label=col_b, color="tomato", linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.suptitle("Training History — ConDetection-DANN", fontweight="bold")
        fig.tight_layout()
        _save(fig, out_path)


def plot_roc_curves(
    results: dict[str, dict],
    out_path: str,
    title: str = "ROC Curves",
) -> None:
    """Overlay ROC curves for multiple models on one axis.

    results: {model_name: {"y_true": ..., "y_score": ..., "AUC": ...}}
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

        for (name, m), color in zip(results.items(), colors):
            if m.get("y_true") is None:
                continue
            fpr, tpr, _ = roc_curve(m["y_true"], m["y_score"])
            auc = m.get("AUC", float("nan"))
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        _save(fig, out_path)


def plot_confusion_matrices(
    results: dict[str, dict],
    out_path: str,
    threshold: float = 0.5,
) -> None:
    """Plot normalised confusion matrices for each model in a grid."""
    with plt.rc_context(STYLE):
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (name, m) in zip(axes, results.items()):
            if m.get("y_true") is None:
                ax.set_visible(False)
                continue
            y_pred = (np.asarray(m["y_score"]) >= threshold).astype(int)
            cm = confusion_matrix(m["y_true"], y_pred, normalize="true")
            disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(name)

        fig.suptitle("Confusion Matrices (normalised)", fontweight="bold")
        fig.tight_layout()
        _save(fig, out_path)


def plot_sota_comparison(df: pd.DataFrame, out_path: str) -> None:
    """Bar chart comparing EER and AUC across all models."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        models = df["model"].tolist()
        x = np.arange(len(models))
        w = 0.35

        colors_for = "#4C72B0"
        colors_itw = "#DD8452"
        highlight = "#2ca02c"  # ConDetection highlight

        bar_colors_for = [highlight if m == "condetection" else colors_for for m in models]
        bar_colors_itw = [highlight if m == "condetection" else colors_itw for m in models]

        # EER plot (lower is better)
        ax = axes[0]
        ax.bar(x - w / 2, df["for_eer"], w, label="FoR Test", color=bar_colors_for, alpha=0.85)
        ax.bar(x + w / 2, df["itw_eer"], w, label="In-the-Wild", color=bar_colors_itw, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("EER (lower is better)")
        ax.set_title("Equal Error Rate")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # AUC plot (higher is better)
        ax = axes[1]
        ax.bar(x - w / 2, df["for_auc"], w, label="FoR Test", color=bar_colors_for, alpha=0.85)
        ax.bar(x + w / 2, df["itw_auc"], w, label="In-the-Wild", color=bar_colors_itw, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("AUC (higher is better)")
        ax.set_title("Area Under ROC Curve")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.4, 1.02)

        fig.suptitle("SOTA Comparative Study", fontweight="bold", fontsize=14)
        fig.tight_layout()
        _save(fig, out_path)


def plot_ablation_chart(df: pd.DataFrame, out_path: str) -> None:
    """Horizontal bar chart showing ablation ITW EER (lower is better)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.8)))
        colors = ["#2ca02c" if n == "full" else "#4C72B0" for n in df["name"]]
        bars = ax.barh(df["description"], df["itw_eer"], color=colors, alpha=0.85)

        for rect, val in zip(bars, df["itw_eer"]):
            ax.text(
                val + 0.002,
                rect.get_y() + rect.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left",
                fontsize=10,
            )

        ax.set_xlabel("ITW EER (lower is better)")
        ax.set_title("Ablation Study — Component Contribution", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        fig.tight_layout()
        _save(fig, out_path)


def plot_score_distributions(
    results: dict[str, dict],
    out_path: str,
) -> None:
    """Score distribution histograms: real vs fake separation per dataset."""
    with plt.rc_context(STYLE):
        n_datasets = len(results)
        fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 5))
        if n_datasets == 1:
            axes = [axes]

        for ax, (name, m) in zip(axes, results.items()):
            y_t = np.asarray(m["y_true"])
            y_s = np.asarray(m["y_score"])
            ax.hist(y_s[y_t == 0], bins=50, alpha=0.6, label="Real", color="steelblue", density=True)
            ax.hist(y_s[y_t == 1], bins=50, alpha=0.6, label="Fake", color="tomato", density=True)
            ax.set_title(f"Score Distribution — {name}")
            ax.set_xlabel("Predicted Score")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.suptitle("Real vs Fake Score Separation", fontweight="bold")
        fig.tight_layout()
        _save(fig, out_path)


def plot_generalization_gap(
    model_results: dict[str, tuple[float, float]],
    out_path: str,
) -> None:
    """Muller-style generalization gap chart.

    model_results: {model_name: (for_eer, itw_eer)}
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(model_results))
        names = list(model_results.keys())
        for_eers = [model_results[n][0] for n in names]
        itw_eers = [model_results[n][1] for n in names]
        gaps = [b - a for a, b in zip(for_eers, itw_eers)]

        ax.bar(x, for_eers, label="FoR Test (in-domain)", color="#4C72B0", alpha=0.85)
        ax.bar(x, gaps, bottom=for_eers, label="Generalization Gap (+ITW)", color="#DD8452", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("EER")
        ax.set_title("Cross-Domain Generalization Gap (Muller et al. style)", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, out_path)
