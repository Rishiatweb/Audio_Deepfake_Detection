"""Ablation study runner: train multiple model variants and compare."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.config import Config
from src.models.factory import get_model
from src.training.losses import build_criterion
from src.training.scheduler import get_cosine_schedule_with_warmup
from src.training.trainer import evaluate, find_best_threshold, save_checkpoint, train_one_epoch


@dataclass
class AblationConfig:
    name: str
    description: str
    config_overrides: dict  # field-path overrides, e.g. {"dann.enabled": False}


# Predefined ablation experiments
ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig(
        name="full",
        description="Full ConDetection-DANN (all components)",
        config_overrides={},
    ),
    AblationConfig(
        name="no_dann",
        description="No DANN (domain discriminator disabled)",
        config_overrides={"dann.enabled": False},
    ),
    AblationConfig(
        name="single_scale_mid",
        description="Single scale (mid only)",
        config_overrides={"mel_configs": "mid_only"},
    ),
    AblationConfig(
        name="no_mixstyle",
        description="No MixStyle domain augmentation",
        config_overrides={"domain_gen.mixstyle_p": 0.0},
    ),
    AblationConfig(
        name="no_consistency",
        description="No consistency loss (lambda_c=0)",
        config_overrides={"training.lambda_c": 0.0},
    ),
]


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Apply dot-path overrides to a copy of config."""
    cfg = copy.deepcopy(cfg)
    _SPECIAL_KEYS = {"mel_configs"}  # handled separately below

    for key, value in overrides.items():
        if key in _SPECIAL_KEYS:
            continue
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    # Special override: single mid-scale only
    if overrides.get("mel_configs") == "mid_only":
        cfg.mel_configs = [cfg.mel_configs[1]]  # keep only mid-resolution mel config

    return cfg


def run_ablation_experiment(
    ablation: AblationConfig,
    base_cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    for_test_loader: DataLoader,
    itw_loader: DataLoader,
    device: torch.device,
    itw_train_loader: DataLoader | None = None,
) -> dict:
    """Train one ablation variant and return evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"Ablation: {ablation.name} — {ablation.description}")
    print("=" * 60)

    cfg = apply_overrides(base_cfg, ablation.config_overrides)
    ckpt_dir = Path(cfg.paths.checkpoint_dir) / f"ablation_{ablation.name}"
    ckpt_path = ckpt_dir / "model_best.pt"
    results_cache = ckpt_dir / "eval_results.json"

    # Check cache BEFORE building model (avoid wasting VRAM on cached variants)
    if ckpt_path.exists() and results_cache.exists():
        print("  Found existing checkpoint — skipping training.")
        with open(results_cache) as f:
            cached = json.load(f)
        # Backfill disc_acc fields missing from old cached results
        for field in ("disc_acc_final", "disc_acc_mean", "disc_acc_deviation"):
            cached.setdefault(field, float("nan"))
        return cached

    model = get_model("condetection", cfg).to(device)

    n_real = sum(1 for lbl in train_loader.dataset.labels if lbl == 0.0)  # type: ignore[attr-defined]
    n_fake = sum(1 for lbl in train_loader.dataset.labels if lbl == 1.0)  # type: ignore[attr-defined]
    criterion = build_criterion(cfg, device, pos_weight_val=n_real / max(n_fake, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=1e-2)

    effective_steps = len(train_loader) if cfg.training.max_train_steps is None else cfg.training.max_train_steps
    total_steps = cfg.training.epochs * effective_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    best_eer = 1.0
    best_threshold = 0.5
    patience_cnt = 0
    disc_acc_history: list[float] = []  # per-epoch disc_acc (NaN when DANN inactive)

    for epoch in range(1, cfg.training.epochs + 1):
        itw_iter = iter(itw_train_loader) if (itw_train_loader and cfg.dann.enabled) else None
        _, disc_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device, cfg, epoch, itw_iter)
        disc_acc_history.append(disc_acc)

        max_val = cfg.training.max_val_steps
        val_raw = evaluate(model, val_loader, criterion, device, cfg, threshold=0.5, max_steps=max_val)
        auc = val_raw.get("AUC", float("nan"))
        if isinstance(auc, float) and not np.isnan(auc):
            val_thr, _ = find_best_threshold(val_raw["y_true"], val_raw["y_score"])
        else:
            val_thr = 0.5

        val_m = evaluate(model, val_loader, criterion, device, cfg, threshold=val_thr, max_steps=max_val)
        eer = val_m["EER"]
        if isinstance(eer, float) and not np.isnan(eer):  # valid
            if eer < best_eer:
                best_eer = eer
                best_threshold = val_thr
                patience_cnt = 0
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_eer,
                    best_threshold,
                    f"{cfg.paths.checkpoint_dir}/ablation_{ablation.name}",
                )
            else:
                patience_cnt += 1

        disc_str = f" | disc_acc={disc_acc:.3f} (|dev|={abs(disc_acc - 0.5):.3f})" if not np.isnan(disc_acc) else ""
        print(f"  Ep {epoch:02d} | val EER={eer:.4f} | AUC={val_m['AUC']:.4f}{disc_str}")
        if patience_cnt >= cfg.training.patience:
            print("  Early stopping.")
            break

    # Summarise disc_acc across training run
    valid_disc = [d for d in disc_acc_history if not np.isnan(d)]
    disc_acc_final = valid_disc[-1] if valid_disc else float("nan")
    disc_acc_mean = float(np.mean(valid_disc)) if valid_disc else float("nan")
    disc_acc_deviation = abs(disc_acc_final - 0.5) if not np.isnan(disc_acc_final) else float("nan")

    # Final evaluation with best threshold
    for_m = evaluate(model, for_test_loader, criterion, device, cfg, threshold=best_threshold)
    itw_m = evaluate(model, itw_loader, criterion, device, cfg, threshold=best_threshold)

    result = {
        "name": ablation.name,
        "description": ablation.description,
        "for_eer": for_m["EER"],
        "for_auc": for_m["AUC"],
        "for_f1": for_m["F1"],
        "itw_eer": itw_m["EER"],
        "itw_auc": itw_m["AUC"],
        "itw_f1": itw_m["F1"],
        "gen_gap_eer": (
            itw_m["EER"] - for_m["EER"] if not any(np.isnan(v) for v in [for_m["EER"], itw_m["EER"]]) else float("nan")
        ),
        # Domain discriminator accuracy (DANN effectiveness metric)
        # Target: 0.5 — discriminator at chance = domain-invariant features learned
        # NaN for no_dann variant (discriminator disabled)
        "disc_acc_final": disc_acc_final,
        "disc_acc_mean": disc_acc_mean,
        "disc_acc_deviation": disc_acc_deviation,  # |final - 0.5|, lower = better DANN
    }
    # Cache eval results so reruns skip this variant
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(results_cache, "w") as f:
        json.dump(result, f)
    return result


def run_all_ablations(
    base_cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    for_test_loader: DataLoader,
    itw_loader: DataLoader,
    device: torch.device,
    itw_train_loader: DataLoader | None = None,
    ablation_names: list[str] | None = None,
) -> pd.DataFrame:
    """Run all (or selected) ablation experiments and return results DataFrame."""
    configs = ABLATION_CONFIGS
    if ablation_names:
        configs = [a for a in configs if a.name in ablation_names]

    results = []
    for abl in configs:
        result = run_ablation_experiment(
            abl, base_cfg, train_loader, val_loader, for_test_loader, itw_loader, device, itw_train_loader
        )
        results.append(result)

    df = pd.DataFrame(results)
    print("\nAblation Results:")
    print(df.to_string(index=False))
    return df
