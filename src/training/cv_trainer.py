"""K-fold cross-validation trainer for ConDetection-DANN."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.config import Config
from src.data.datasets import FastAudioDataset, source_balanced_cap, stratified_cap
from src.models.factory import get_model
from src.training.losses import build_criterion
from src.training.scheduler import get_cosine_schedule_with_warmup
from src.training.trainer import evaluate, kfold_calibrate_threshold, save_checkpoint, train_one_epoch


def run_kfold_cv(
    cfg: Config,
    model_name: str,
    device: torch.device,
) -> pd.DataFrame:
    """K-fold cross-validation on merged FoR train+val data.

    Uses CrossValConfig: n_folds, cv_epochs, cv_max_train_steps.
    Returns per-fold metrics DataFrame; prints mean ± std summary.
    """
    from src.data.datasets import build_splits

    cv_cfg = copy.deepcopy(cfg)
    cv_cfg.training.epochs = cfg.cross_val.cv_epochs
    if cfg.cross_val.cv_max_train_steps is not None:
        cv_cfg.training.max_train_steps = cfg.cross_val.cv_max_train_steps

    train_df, val_df, _, _ = build_splits(cfg)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    n_folds = cfg.cross_val.n_folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.training.seed)

    fold_results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_val_df, train_val_df["label"].astype(int)), 1
    ):
        print(f"\n{'=' * 60}")
        print(f"K-fold CV — Fold {fold}/{n_folds}")
        print("=" * 60)

        fold_train_df = source_balanced_cap(
            train_val_df.iloc[train_idx].reset_index(drop=True),
            cfg.training.max_train_samples,
            cfg.training.seed,
        )
        fold_val_df = stratified_cap(
            train_val_df.iloc[val_idx].reset_index(drop=True),
            cfg.training.max_val_samples,
            cfg.training.seed,
        )
        print(f"  Fold train: {len(fold_train_df):,}  Fold val: {len(fold_val_df):,}")

        worker_kw: dict = dict(
            num_workers=cfg.training.num_workers,
            pin_memory=(cfg.training.num_workers > 0),
            persistent_workers=(cfg.training.num_workers > 0),
        )
        fold_train_loader = DataLoader(
            FastAudioDataset(fold_train_df, augment=True, cfg=cfg),
            batch_size=cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            **worker_kw,
        )
        fold_val_loader = DataLoader(
            FastAudioDataset(fold_val_df, augment=False, cfg=cfg),
            batch_size=cfg.training.batch_size * 2,
            shuffle=False,
            drop_last=False,
            **worker_kw,
        )

        model = get_model(model_name, cv_cfg).to(device)

        n_real = int((fold_train_df["label"] == 0).sum())
        n_fake = int((fold_train_df["label"] == 1).sum())
        criterion = build_criterion(cv_cfg, device, pos_weight_val=n_real / max(n_fake, 1))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cv_cfg.training.learning_rate, weight_decay=1e-2
        )

        effective = (
            len(fold_train_loader)
            if cv_cfg.training.max_train_steps is None
            else cv_cfg.training.max_train_steps
        )
        total_steps = cv_cfg.training.epochs * effective
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, int(0.1 * total_steps), total_steps
        )
        scaler = GradScaler(device.type, enabled=device.type == "cuda")

        best_eer = 1.0
        best_thr = 0.5
        patience_cnt = 0

        for epoch in range(1, cv_cfg.training.epochs + 1):
            train_one_epoch(
                model, fold_train_loader, optimizer, scheduler, scaler,
                criterion, device, cv_cfg, epoch, None,
            )
            val_raw = evaluate(
                model, fold_val_loader, criterion, device, cv_cfg,
                threshold=0.5, max_steps=cv_cfg.training.max_val_steps,
            )
            auc = val_raw.get("AUC", float("nan"))
            val_thr = (
                kfold_calibrate_threshold(val_raw["y_true"], val_raw["y_score"])
                if not np.isnan(float(auc))
                else 0.5
            )
            val_m = evaluate(
                model, fold_val_loader, criterion, device, cv_cfg,
                threshold=val_thr, max_steps=cv_cfg.training.max_val_steps,
            )
            eer = val_m["EER"]
            if not np.isnan(float(eer)) and eer < best_eer:
                best_eer = eer
                best_thr = val_thr
                patience_cnt = 0
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, best_eer, best_thr,
                    f"{cfg.paths.checkpoint_dir}/cv_fold{fold}",
                )
            else:
                patience_cnt += 1

            print(f"  Ep {epoch:02d} | val EER={eer:.4f} | AUC={val_m['AUC']:.4f}")
            if patience_cnt >= cv_cfg.training.patience:
                print("  Early stopping.")
                break

        # Final evaluation on held-out fold val
        final_val = evaluate(model, fold_val_loader, criterion, device, cv_cfg, threshold=best_thr)
        fold_results.append(
            {
                "fold": fold,
                "val_eer": final_val["EER"],
                "val_auc": final_val["AUC"],
                "val_f1": final_val["F1"],
                "val_acc": final_val["Acc"],
                "best_threshold": best_thr,
            }
        )
        print(
            f"Fold {fold} done | EER={final_val['EER']:.4f} | AUC={final_val['AUC']:.4f} | "
            f"F1={final_val['F1']:.4f}"
        )

    df = pd.DataFrame(fold_results)
    print(f"\n{'=' * 60}")
    print(f"K-fold CV Summary ({n_folds} folds, model={model_name})")
    print("=" * 60)
    for col in ["val_eer", "val_auc", "val_f1"]:
        print(f"  {col}: {df[col].mean():.4f} ± {df[col].std():.4f}")
    return df
