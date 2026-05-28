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


def plot_cross_scale_attention_heatmap(
    attn_weights,
    scale_names: list[str],
    out_path: str,
    title: str = "Cross-Scale Attention Weights",
) -> None:
    """Visualise CrossScaleAttentionFusion weights as a heatmap.

    attn_weights: (B, K, K) or (K, K) tensor or ndarray — output of
    CrossScaleAttentionFusion, averaged over batch if 3-D.
    Rows = query scales, columns = key scales.
    """
    if hasattr(attn_weights, "detach"):
        attn_weights = attn_weights.detach().cpu().numpy()
    attn = np.asarray(attn_weights, dtype=np.float32)
    if attn.ndim == 3:
        attn = attn.mean(axis=0)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(attn, cmap="Blues", vmin=0.0, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(scale_names)))
        ax.set_yticks(range(len(scale_names)))
        ax.set_xticklabels(scale_names)
        ax.set_yticklabels(scale_names)
        ax.set_xlabel("Key Scale")
        ax.set_ylabel("Query Scale")
        ax.set_title(title, fontweight="bold")
        thresh = attn.max() * 0.6 if attn.max() > 0 else 1.0
        for i in range(len(scale_names)):
            for j in range(len(scale_names)):
                color = "white" if attn[i, j] > thresh else "black"
                ax.text(j, i, f"{attn[i, j]:.3f}", ha="center", va="center",
                        fontsize=9, color=color)
        fig.tight_layout()
        _save(fig, out_path)


def plot_ablation_metrics_grouped(df: pd.DataFrame, out_path: str) -> None:
    """Grouped bar chart: EER + AUC on FoR and ITW per ablation variant."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        names = df["name"].tolist()
        x = np.arange(len(names))
        w = 0.35

        highlight = "#2ca02c"
        c_for = "#4C72B0"
        c_itw = "#DD8452"
        colors_for = [highlight if n == "full" else c_for for n in names]
        colors_itw = [highlight if n == "full" else c_itw for n in names]

        # EER (lower better)
        ax = axes[0]
        ax.bar(x - w / 2, df["for_eer"], w, label="FoR Test", color=colors_for, alpha=0.85)
        ax.bar(x + w / 2, df["itw_eer"], w, label="In-the-Wild", color=colors_itw, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("EER (lower is better)")
        ax.set_title("Equal Error Rate per Ablation")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # AUC (higher better)
        ax = axes[1]
        ax.bar(x - w / 2, df["for_auc"], w, label="FoR Test", color=colors_for, alpha=0.85)
        ax.bar(x + w / 2, df["itw_auc"], w, label="In-the-Wild", color=colors_itw, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("AUC (higher is better)")
        ax.set_title("Area Under ROC per Ablation")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.5, 1.02)

        fig.suptitle("Ablation Study — EER & AUC Comparison", fontweight="bold", fontsize=14)
        fig.tight_layout()
        _save(fig, out_path)


def plot_ablation_gen_gap(df: pd.DataFrame, out_path: str) -> None:
    """Stacked bar: FoR EER (base) + generalization gap (delta) per ablation."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        names = df["name"].tolist()
        x = np.arange(len(names))

        for_eers = df["for_eer"].to_numpy()
        gaps = df["gen_gap_eer"].to_numpy()
        colors_base = ["#2ca02c" if n == "full" else "#4C72B0" for n in names]

        ax.bar(x, for_eers, label="FoR Test EER (in-domain)", color=colors_base, alpha=0.85)
        ax.bar(x, gaps, bottom=for_eers, label="Generalization Gap (+ITW)", color="#DD8452", alpha=0.85)

        # Value labels on top
        for i, (fe, g) in enumerate(zip(for_eers, gaps)):
            ax.text(i, fe + g + 0.005, f"{fe + g:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("EER")
        ax.set_title("Ablation — Generalization Gap (FoR → ITW)", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, out_path)


def plot_ablation_disc_acc(df: pd.DataFrame, out_path: str) -> None:
    """Bar chart of domain discriminator accuracy deviation |disc_acc - 0.5| per ablation.

    Lower = better DANN (discriminator at chance = domain-invariant features).
    Variants without DANN (disc_acc=NaN) shown as hatched gray.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        names = df["name"].tolist()
        x = np.arange(len(names))

        deviations = df["disc_acc_deviation"].to_numpy()
        disc_finals = df["disc_acc_final"].to_numpy()

        colors = []
        hatches = []
        for i, n in enumerate(names):
            if np.isnan(deviations[i]):
                colors.append("#CCCCCC")
                hatches.append("///")
            elif n == "full":
                colors.append("#2ca02c")
                hatches.append("")
            else:
                colors.append("#4C72B0")
                hatches.append("")

        bars = ax.bar(x, [d if not np.isnan(d) else 0.0 for d in deviations],
                       color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)

        # Value labels
        for i, (d, da) in enumerate(zip(deviations, disc_finals)):
            if np.isnan(d):
                ax.text(i, 0.01, "N/A\n(no DANN)", ha="center", va="bottom", fontsize=8, color="gray")
            else:
                ax.text(i, d + 0.003, f"|{da:.3f} − 0.5|\n= {d:.3f}", ha="center", va="bottom", fontsize=8)

        ax.axhline(y=0.0, color="green", linestyle="--", alpha=0.5, label="Ideal (0.0 = perfect DANN)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("|disc_acc − 0.5| (lower = better)")
        ax.set_title("Domain Discriminator Accuracy Deviation per Ablation", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        _save(fig, out_path)


def plot_ablation_f1_comparison(df: pd.DataFrame, out_path: str) -> None:
    """Grouped bar: F1 score on FoR vs ITW per ablation."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        names = df["name"].tolist()
        x = np.arange(len(names))
        w = 0.35

        highlight = "#2ca02c"
        c_for = "#4C72B0"
        c_itw = "#DD8452"
        colors_for = [highlight if n == "full" else c_for for n in names]
        colors_itw = [highlight if n == "full" else c_itw for n in names]

        ax.bar(x - w / 2, df["for_f1"], w, label="FoR Test", color=colors_for, alpha=0.85)
        ax.bar(x + w / 2, df["itw_f1"], w, label="In-the-Wild", color=colors_itw, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("F1 Score")
        ax.set_title("Ablation — F1 Score Comparison", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        _save(fig, out_path)


def plot_ablation_radar(df: pd.DataFrame, out_path: str) -> None:
    """Radar/spider chart comparing ablation variants across key metrics."""
    with plt.rc_context(STYLE):
        metrics = ["for_eer", "itw_eer", "for_auc", "itw_auc", "for_f1", "itw_f1"]
        metric_labels = ["FoR EER↓", "ITW EER↓", "FoR AUC↑", "ITW AUC↑", "FoR F1↑", "ITW F1↑"]

        # Normalize: for EER (lower=better) invert, for AUC/F1 keep as-is
        names = df["name"].tolist()
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(names)))

        for idx, (_, row) in enumerate(df.iterrows()):
            values = []
            for m in metrics:
                v = row[m]
                if "eer" in m:
                    values.append(1.0 - v)  # invert EER so higher = better
                else:
                    values.append(v)
            values += values[:1]
            lw = 3 if row["name"] == "full" else 1.5
            ax.plot(angles, values, "o-", linewidth=lw, label=row["name"], color=colors[idx])
            ax.fill(angles, values, alpha=0.08, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title("Ablation — Multi-Metric Radar", fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        _save(fig, out_path)


def plot_ablation_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Heatmap of all ablation metrics — quick visual overview."""
    with plt.rc_context(STYLE):
        cols = ["for_eer", "itw_eer", "for_auc", "itw_auc", "for_f1", "itw_f1", "gen_gap_eer"]
        labels = ["FoR EER↓", "ITW EER↓", "FoR AUC↑", "ITW AUC↑", "FoR F1↑", "ITW F1↑", "Gen Gap↓"]
        data = df[cols].to_numpy().astype(float)
        names = df["name"].tolist()

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(names)

        for i in range(len(names)):
            for j in range(len(labels)):
                val = data[i, j]
                if np.isnan(val):
                    txt = "N/A"
                else:
                    txt = f"{val:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if val > 0.5 else "black")

        ax.set_title("Ablation Metrics Heatmap", fontweight="bold")
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
