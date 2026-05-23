"""SOTA comparative study: train all baselines and compare with ConDetection-DANN."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from src.config import load_config
from src.data.datasets import FastAudioDataset, build_splits, make_loaders
from src.evaluation.comparative import run_comparative_study
from src.models.factory import get_model
from src.training.trainer import find_best_threshold, evaluate
from src.training.losses import build_criterion


def parse_args():
    p = argparse.ArgumentParser(description="Run SOTA comparative study")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--condetection-ckpt", default=None, help="Pre-trained ConDetection checkpoint (.pt)")
    p.add_argument("--models", nargs="+", default=None, help="Subset of models to run")
    p.add_argument("--output", default=None, help="Output CSV path for results table")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df, val_df, for_test_df, itw_df = build_splits(cfg)
    train_loader, val_loader, for_test_loader, itw_loader = make_loaders(train_df, val_df, for_test_df, itw_df, cfg)

    itw_train_loader = None
    if cfg.dann.enabled and len(itw_df) > 0:
        itw_sample = itw_df.sample(min(cfg.training.max_train_samples, len(itw_df)), random_state=cfg.training.seed).reset_index(drop=True)
        itw_train_ds = FastAudioDataset(itw_sample, augment=True, cfg=cfg)
        itw_train_loader = torch.utils.data.DataLoader(
            itw_train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0, drop_last=True
        )

    # Load pre-trained ConDetection if provided
    cd_model = None
    cd_threshold = 0.5
    if args.condetection_ckpt:
        cd_model = get_model("condetection", cfg).to(device)
        cd_model.load_state_dict(torch.load(args.condetection_ckpt, map_location=device, weights_only=True))
        criterion = build_criterion(cfg, device)
        val_m = evaluate(cd_model, val_loader, criterion, device, cfg, threshold=0.5)
        if val_m["AUC"] == val_m["AUC"]:
            cd_threshold, _ = find_best_threshold(val_m["y_true"], val_m["y_score"])
        print(f"Loaded ConDetection checkpoint: {args.condetection_ckpt}")
        print(f"  Val EER={val_m['EER']:.4f} | AUC={val_m['AUC']:.4f} | threshold={cd_threshold:.3f}")

    df = run_comparative_study(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        for_test_loader=for_test_loader,
        itw_loader=itw_loader,
        device=device,
        condetection_model=cd_model,
        condetection_threshold=cd_threshold,
        itw_train_loader=itw_train_loader,
        model_names=args.models,
    )

    out_path = args.output or f"{cfg.paths.tables_dir}/comparative_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
