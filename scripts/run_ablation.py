"""Run ablation studies."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from src.config import load_config
from src.data.datasets import FastAudioDataset, build_splits, make_loaders
from src.training.ablation import run_all_ablations


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation studies")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ablations", nargs="+", default=None, help="Subset of ablation names to run")
    p.add_argument("--output", default=None, help="Output CSV path")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs per variant")
    p.add_argument("--patience", type=int, default=None, help="Override early stopping patience per variant")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.make_dirs()
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.patience is not None:
        cfg.training.patience = args.patience

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

    df = run_all_ablations(
        base_cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        for_test_loader=for_test_loader,
        itw_loader=itw_loader,
        device=device,
        itw_train_loader=itw_train_loader,
        ablation_names=args.ablations,
    )

    out_path = args.output or f"{cfg.paths.tables_dir}/ablation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Ablation results saved to {out_path}")


if __name__ == "__main__":
    main()
