"""Generate all ablation study figures from saved results."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd

from src.visualization.figures import (
    plot_ablation_chart,
    plot_ablation_disc_acc,
    plot_ablation_f1_comparison,
    plot_ablation_gen_gap,
    plot_ablation_heatmap,
    plot_ablation_metrics_grouped,
    plot_ablation_radar,
)


def main():
    ablation_csv = "results/tables/ablation_results.csv"
    if not Path(ablation_csv).exists():
        print(f"ERROR: {ablation_csv} not found. Run ablation study first.")
        sys.exit(1)

    df = pd.read_csv(ablation_csv)
    print(f"Loaded {len(df)} ablation variants from {ablation_csv}")
    print(df[["name", "for_eer", "itw_eer", "gen_gap_eer"]].to_string(index=False))

    out_dir = Path("results/ablation_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. ITW EER horizontal bar (existing function)
    plot_ablation_chart(df, str(out_dir / "ablation_itw_eer.png"))
    print("Saved ablation_itw_eer.png")

    # 2. Grouped EER + AUC bars (FoR vs ITW)
    plot_ablation_metrics_grouped(df, str(out_dir / "ablation_eer_auc_grouped.png"))
    print("Saved ablation_eer_auc_grouped.png")

    # 3. Generalization gap stacked bar
    plot_ablation_gen_gap(df, str(out_dir / "ablation_gen_gap.png"))
    print("Saved ablation_gen_gap.png")

    # 4. Domain discriminator accuracy deviation
    plot_ablation_disc_acc(df, str(out_dir / "ablation_disc_acc.png"))
    print("Saved ablation_disc_acc.png")

    # 5. F1 comparison
    plot_ablation_f1_comparison(df, str(out_dir / "ablation_f1_comparison.png"))
    print("Saved ablation_f1_comparison.png")

    # 6. Radar chart (multi-metric)
    plot_ablation_radar(df, str(out_dir / "ablation_radar.png"))
    print("Saved ablation_radar.png")

    # 7. Metrics heatmap
    plot_ablation_heatmap(df, str(out_dir / "ablation_heatmap.png"))
    print("Saved ablation_heatmap.png")

    print(f"\nAll 7 ablation figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
