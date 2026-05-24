"""Standalone evaluation: load a checkpoint and evaluate on FoR test + ITW."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import torch

from src.config import load_config
from src.data.datasets import FastAudioDataset, build_splits
from src.models.factory import count_parameters, get_model
from src.training.losses import build_criterion
from src.training.trainer import evaluate, evaluate_tta, find_best_threshold


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model on FoR test + ITW")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--ckpt", required=True, help="Model checkpoint .pt file")
    p.add_argument("--model", default="condetection", help="Model name")
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold (default: from checkpoint dir or val grid search)")
    p.add_argument("--no-tta", action="store_true", help="Disable TTA even if config enables it")
    p.add_argument("--output", default=None, help="Save metrics CSV to this path")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = get_model(args.model, cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()
    params = count_parameters(model)
    print(f"Loaded: {args.ckpt}  ({params['trainable']:,} params)")

    # Build test loaders (no train needed)
    _, val_df, for_test_df, itw_df = build_splits(cfg)
    worker_kw: dict = dict(
        num_workers=cfg.training.num_workers,
        pin_memory=(cfg.training.num_workers > 0),
        persistent_workers=(cfg.training.num_workers > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        FastAudioDataset(val_df, augment=False, cfg=cfg),
        batch_size=cfg.training.batch_size * 2, shuffle=False, drop_last=False, **worker_kw,
    )
    for_test_loader = torch.utils.data.DataLoader(
        FastAudioDataset(for_test_df, augment=False, cfg=cfg),
        batch_size=cfg.training.batch_size * 2, shuffle=False, drop_last=False, **worker_kw,
    )
    itw_loader = torch.utils.data.DataLoader(
        FastAudioDataset(itw_df, augment=False, cfg=cfg),
        batch_size=cfg.training.batch_size * 2, shuffle=False, drop_last=False, **worker_kw,
    )

    # Determine threshold
    threshold = args.threshold
    if threshold is None:
        thr_file = Path(args.ckpt).parent / "best_threshold.txt"
        if thr_file.exists():
            threshold = float(thr_file.read_text(encoding="utf-8").strip())
            print(f"Threshold (from file): {threshold:.4f}")
        else:
            criterion_tmp = build_criterion(cfg, device)
            val_m = evaluate(model, val_loader, criterion_tmp, device, cfg,
                             threshold=0.5, max_steps=cfg.training.max_val_steps)
            if val_m["AUC"] == val_m["AUC"]:
                threshold, _ = find_best_threshold(val_m["y_true"], val_m["y_score"])
            else:
                threshold = 0.5
            print(f"Threshold (val grid search): {threshold:.4f}")

    use_tta = cfg.training.use_tta and not args.no_tta
    eval_fn = evaluate_tta if use_tta else evaluate
    tta_kwargs = {"tta_shifts": tuple(cfg.training.tta_shifts)} if use_tta else {}
    print(f"TTA: {'enabled' if use_tta else 'disabled'}")

    criterion = build_criterion(cfg, device)
    print("\nEvaluating...")
    for_m = eval_fn(model, for_test_loader, criterion, device, cfg,
                    threshold=threshold, **tta_kwargs)
    itw_m = eval_fn(model, itw_loader, criterion, device, cfg,
                    threshold=threshold, **tta_kwargs)

    print("\n" + "=" * 55)
    print(f"  FoR Test | EER={for_m['EER']:.4f} | AUC={for_m['AUC']:.4f} | "
          f"F1={for_m['F1']:.4f} | Acc={for_m['Acc']:.4f}")
    print(f"  ITW      | EER={itw_m['EER']:.4f} | AUC={itw_m['AUC']:.4f} | "
          f"F1={itw_m['F1']:.4f} | Acc={itw_m['Acc']:.4f}")
    if for_m["EER"] == for_m["EER"] and itw_m["EER"] == itw_m["EER"]:
        print(f"  Gen Gap  | DEER={itw_m['EER'] - for_m['EER']:+.4f}")
    print("=" * 55)

    if args.output:
        rows = [
            {"dataset": "FoR Test", **{k: v for k, v in for_m.items() if not k.startswith("y_")}},
            {"dataset": "ITW", **{k: v for k, v in itw_m.items() if not k.startswith("y_")}},
        ]
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
