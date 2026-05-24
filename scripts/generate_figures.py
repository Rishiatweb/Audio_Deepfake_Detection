"""Generate all publication figures from saved results/checkpoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import torch

from src.config import load_config
from src.data.datasets import build_splits, make_loaders
from src.models.factory import get_model
from src.training.losses import build_criterion
from src.training.trainer import evaluate
from src.visualization.figures import (
    plot_ablation_chart,
    plot_confusion_matrices,
    plot_cross_scale_attention_heatmap,
    plot_generalization_gap,
    plot_roc_curves,
    plot_score_distributions,
    plot_sota_comparison,
    plot_training_curves,
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate all paper figures")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", default=None, help="ConDetection checkpoint .pt")
    p.add_argument("--history", default=None, help="training_history.csv path")
    p.add_argument("--comparison", default=None, help="comparative_results.csv path")
    p.add_argument("--ablation", default=None, help="ablation_results.csv path")
    p.add_argument("--tsne", action="store_true", help="Generate t-SNE (slow)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_dir = cfg.paths.figures_dir

    # ─── Training curves ───
    history_path = args.history or f"{cfg.paths.output_dir}/training_history.csv"
    if Path(history_path).exists():
        history = pd.read_csv(history_path).to_dict("records")
        plot_training_curves(history, f"{fig_dir}/training_curves.png")
        print("Saved training_curves.png")

    # ─── Model evaluation figures ───
    ckpt_path = args.ckpt or f"{cfg.paths.checkpoint_dir}/model_best.pt"
    if Path(ckpt_path).exists():
        print("Loading model and data for evaluation figures...")
        model = get_model("condetection", cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        _, _, for_test_df, itw_df = build_splits(cfg)
        _, _, for_test_loader, itw_loader = make_loaders(
            pd.DataFrame(), pd.DataFrame(), for_test_df, itw_df, cfg
        )

        # Find best threshold from val if available
        threshold_file = Path(cfg.paths.checkpoint_dir) / "best_threshold.txt"
        threshold = 0.5
        if threshold_file.exists():
            threshold = float(threshold_file.read_text().strip())

        criterion = build_criterion(cfg, device)
        for_m = evaluate(model, for_test_loader, criterion, device, cfg, threshold=threshold)
        itw_m = evaluate(model, itw_loader, criterion, device, cfg, threshold=threshold)

        results = {"FoR Test": for_m, "In-the-Wild": itw_m}

        plot_roc_curves(results, f"{fig_dir}/roc_curves.png", "ROC Curves — ConDetection-DANN")
        plot_confusion_matrices(results, f"{fig_dir}/confusion_matrices.png", threshold=threshold)
        plot_score_distributions(results, f"{fig_dir}/score_distributions.png")
        print("Saved ROC, confusion matrices, score distributions.")

        # Cross-scale attention heatmap
        try:
            m_inner = model.module if hasattr(model, "module") else model
            attn_w = getattr(m_inner, "_last_attn_weights", None)
            if attn_w is not None:
                plot_cross_scale_attention_heatmap(
                    attn_w,
                    [mc.name for mc in cfg.mel_configs],
                    f"{fig_dir}/attention_heatmap.png",
                )
                print("Saved attention_heatmap.png")
        except Exception as e:
            print(f"Attention heatmap skipped: {e}")

        # t-SNE (optional, slow)
        if args.tsne:
            try:
                from src.visualization.tsne import plot_tsne_domains
                # Need no-DANN model for comparison
                cfg_no_dann = load_config(args.config)
                cfg_no_dann.dann.enabled = False
                model_no_dann = get_model("condetection", cfg_no_dann).to(device)
                # Use random weights for no-DANN comparison if no ckpt
                plot_tsne_domains(
                    model, model_no_dann,
                    for_test_loader, itw_loader,
                    device, cfg.mel_configs, cfg.audio.sample_rate,
                    f"{fig_dir}/tsne_domain_separation.png",
                )
                print("Saved tsne_domain_separation.png")
            except Exception as e:
                print(f"t-SNE skipped: {e}")

        # Grad-CAM
        try:
            from src.data.spectrograms import make_multires_logmels
            from src.visualization.gradcam import compute_gradcam, plot_gradcam

            batch_wavs, _ = next(iter(for_test_loader))
            batch_wavs = batch_wavs[:3].to(device)
            mels = make_multires_logmels(batch_wavs, cfg.mel_configs, cfg.audio.sample_rate)
            # Compute per-sample
            cams = compute_gradcam(model, [m[:1] for m in mels], device)
            plot_gradcam(
                [m[:3] for m in mels], cams,
                [mc.name for mc in cfg.mel_configs],
                f"{fig_dir}/gradcam.png",
                title="Grad-CAM — ConDetection-DANN (FoR Test)",
            )
            print("Saved gradcam.png")
        except Exception as e:
            print(f"Grad-CAM skipped: {e}")

    # ─── Comparison figure ───
    comparison_path = args.comparison or f"{cfg.paths.tables_dir}/comparative_results.csv"
    if Path(comparison_path).exists():
        df = pd.read_csv(comparison_path)
        plot_sota_comparison(df, f"{fig_dir}/sota_comparison.png")
        gap_data = {row["model"]: (row["for_eer"], row["itw_eer"]) for _, row in df.iterrows()}
        plot_generalization_gap(gap_data, f"{fig_dir}/generalization_gap.png")
        print("Saved sota_comparison.png, generalization_gap.png")

    # ─── Ablation figure ───
    ablation_path = args.ablation or f"{cfg.paths.tables_dir}/ablation_results.csv"
    if Path(ablation_path).exists():
        df = pd.read_csv(ablation_path)
        plot_ablation_chart(df, f"{fig_dir}/ablation_chart.png")
        print("Saved ablation_chart.png")

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
