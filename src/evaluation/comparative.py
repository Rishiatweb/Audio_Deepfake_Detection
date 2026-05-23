"""SOTA comparative study pipeline."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.config import Config
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.statistical import delong_test, mcnemar_test
from src.models.factory import count_parameters, get_model
from src.training.losses import build_criterion
from src.training.scheduler import get_cosine_schedule_with_warmup
from src.training.trainer import evaluate, find_best_threshold, save_checkpoint, train_one_epoch

BASELINE_MODELS = ["lcnn", "aasist", "rawnet2"]


def train_model(
    model_name: str,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    itw_train_loader: DataLoader | None = None,
) -> tuple[nn.Module, float, float]:
    """Train a model from scratch and return (model, best_eer, best_threshold)."""
    print(f"\nTraining {model_name.upper()} from scratch...")
    model = get_model(model_name, cfg).to(device)
    print(f"  Parameters: {count_parameters(model)['trainable']:,}")

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

    # DANN only for ConDetection; baselines train without domain adversarial loss
    use_dann_save = cfg.dann.enabled
    if model_name != "condetection":
        cfg.dann.enabled = False

    try:
        for epoch in range(1, cfg.training.epochs + 1):
            itw_iter = iter(itw_train_loader) if (itw_train_loader and cfg.dann.enabled) else None
            train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device, cfg, epoch, itw_iter)

            max_val = cfg.training.max_val_steps
            val_raw = evaluate(model, val_loader, criterion, device, cfg, threshold=0.5, max_steps=max_val)
            auc = val_raw.get("AUC", float("nan"))
            if not np.isnan(auc):  # not nan
                val_thr, _ = find_best_threshold(val_raw["y_true"], val_raw["y_score"])
            else:
                val_thr = 0.5

            val_m = evaluate(model, val_loader, criterion, device, cfg, threshold=val_thr, max_steps=max_val)
            eer = val_m["EER"]

            if not np.isnan(eer) and eer < best_eer:  # improving
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
                    f"{cfg.paths.checkpoint_dir}/{model_name}",
                )
            else:
                patience_cnt += 1

            print(f"  Ep {epoch:02d} | EER={eer:.4f} | AUC={val_m['AUC']:.4f}")
            if patience_cnt >= cfg.training.patience:
                print("  Early stopping.")
                break

        # Reload best checkpoint
        ckpt_path = Path(cfg.paths.checkpoint_dir) / model_name / "model_best.pt"
        if ckpt_path.exists():
            m_inner = model.module if hasattr(model, "module") else model
            m_inner.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=True))
    finally:
        cfg.dann.enabled = use_dann_save

    return model, best_eer, best_threshold


def run_comparative_study(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    for_test_loader: DataLoader,
    itw_loader: DataLoader,
    device: torch.device,
    condetection_model: nn.Module | None = None,
    condetection_threshold: float = 0.5,
    itw_train_loader: DataLoader | None = None,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Train all baselines and compare against ConDetection-DANN.

    If condetection_model is provided (pre-trained), skip training ConDetection.
    Returns a DataFrame with all metrics for all models.
    """
    if model_names is None:
        model_names = ["condetection"] + BASELINE_MODELS

    all_results: list[dict] = []
    all_models: dict[str, tuple[nn.Module, float, float]] = {}

    criterion = build_criterion(cfg, device)

    # ConDetection (pre-trained or train fresh)
    if "condetection" in model_names:
        if condetection_model is not None:
            cd_model = condetection_model
            cd_threshold = condetection_threshold
        else:
            cd_model, _, cd_threshold = train_model(
                "condetection", cfg, train_loader, val_loader, device, itw_train_loader
            )
        all_models["condetection"] = (cd_model, 0.0, cd_threshold)

    # Train each baseline
    for name in model_names:
        if name == "condetection":
            continue
        model, best_eer, threshold = train_model(name, cfg, train_loader, val_loader, device)
        all_models[name] = (model, best_eer, threshold)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Evaluate all models
    for name, (model, _, threshold) in all_models.items():
        print(f"\nEvaluating {name.upper()}...")
        for_m = evaluate(model, for_test_loader, criterion, device, cfg, threshold=threshold)
        itw_m = evaluate(model, itw_loader, criterion, device, cfg, threshold=threshold)
        params = count_parameters(model)

        gen_gap = float("nan")
        if for_m["EER"] == for_m["EER"] and itw_m["EER"] == itw_m["EER"]:
            gen_gap = itw_m["EER"] - for_m["EER"]

        result: dict = {
            "model": name,
            "params_M": round(params["trainable"] / 1e6, 2),
            "for_eer": for_m["EER"],
            "for_auc": for_m["AUC"],
            "for_min_dcf": compute_all_metrics(
                for_m["y_true"], for_m["y_score"], threshold=threshold, bootstrap=False
            ).get("MinDCF", float("nan")),
            "for_f1": for_m["F1"],
            "itw_eer": itw_m["EER"],
            "itw_auc": itw_m["AUC"],
            "itw_min_dcf": compute_all_metrics(
                itw_m["y_true"], itw_m["y_score"], threshold=threshold, bootstrap=False
            ).get("MinDCF", float("nan")),
            "itw_f1": itw_m["F1"],
            "gen_gap_eer": gen_gap,
            "y_true_for": for_m["y_true"],
            "y_score_for": for_m["y_score"],
            "y_pred_for": (for_m["y_score"] >= threshold).astype(int),
            "y_true_itw": itw_m["y_true"],
            "y_score_itw": itw_m["y_score"],
            "y_pred_itw": (itw_m["y_score"] >= threshold).astype(int),
        }
        all_results.append(result)

    # Statistical significance vs ConDetection on ITW (primary out-of-domain test)
    if "condetection" in all_models:
        cd_result = next(r for r in all_results if r["model"] == "condetection")
        cd_itw_scores = cd_result["y_score_itw"]
        cd_itw_preds = cd_result["y_pred_itw"]
        cd_itw_true = cd_result["y_true_itw"]

        for result in all_results:
            if result["model"] == "condetection":
                continue
            bl_scores = result["y_score_itw"]
            bl_preds = result["y_pred_itw"]

            # McNemar's test (binary predictions)
            mc = mcnemar_test(cd_itw_true, cd_itw_preds, bl_preds)
            result["mcnemar_chi2"] = mc["chi2"]
            result["mcnemar_p"] = mc["p_value"]
            result["mcnemar_sig"] = mc["significant"]

            # DeLong's test (continuous scores)
            dl = delong_test(cd_itw_true, cd_itw_scores, bl_scores)
            result["delong_z"] = dl["z_stat"]
            result["delong_p"] = dl["p_value"]
            result["delong_sig"] = dl["significant"]

    # Build summary table (drop raw arrays)
    summary_cols = [k for k in all_results[0] if not k.startswith("y_")]
    df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])

    print("\n" + "=" * 80)
    print("COMPARATIVE STUDY RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))

    return df
