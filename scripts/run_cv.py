"""K-fold cross-validation runner for ConDetection-DANN models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from src.config import load_config
from src.training.cv_trainer import run_kfold_cv


def parse_args():
    p = argparse.ArgumentParser(description="K-fold cross-validation")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument(
        "--model",
        default="condetection",
        help="Model name (condetection/aasist/lcnn/rawnet2)",
    )
    p.add_argument("--output", default=None, help="Output CSV path for fold results")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print(
        f"K-fold CV: {cfg.cross_val.n_folds} folds, "
        f"{cfg.cross_val.cv_epochs} epochs/fold, "
        f"model={args.model}"
    )

    df = run_kfold_cv(cfg, args.model, device)

    out = args.output or f"{cfg.paths.tables_dir}/cv_results_{args.model}.csv"
    df.to_csv(out, index=False)
    print(f"\nCV results saved to {out}")


if __name__ == "__main__":
    main()
