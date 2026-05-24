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
from src.training.trainer import (
    evaluate,
    find_best_threshold,
    kfold_calibrate_threshold,
    save_checkpoint,
    train_one_epoch,
)

BASELINE_MODELS = ["lcnn", "aasist", "rawnet2"]
SKLEARN_MODELS = ["lr", "rf"]


def _extract_mel_features(
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    max_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed-length mel features for sklearn classifiers.

    Uses mid-scale mel spectrogram (index 1): mean + std across time → 2*n_mels vector.
    Returns (X, y) arrays ready for sklearn.fit().
    """
    from src.data.spectrograms import make_multires_logmels

    mid_idx = min(1, len(cfg.mel_configs) - 1)
    n_feats = cfg.mel_configs[mid_idx].n_mels * 2
    x_list: list[np.ndarray] = []
    y_list: list[float] = []

    for step, (waveforms, labels) in enumerate(loader, 1):
        waveforms = waveforms.to(device, non_blocking=True)
        with torch.no_grad():
            mels = make_multires_logmels(waveforms, cfg.mel_configs, cfg.audio.sample_rate, train_mode=False)
        mel = mels[mid_idx].squeeze(1)  # (B, n_mels, T)
        feats = torch.cat([mel.mean(dim=2), mel.std(dim=2)], dim=1)  # (B, 2*n_mels)
        x_list.append(feats.cpu().numpy())
        y_list.extend(labels.numpy().tolist())
        if max_steps is not None and step >= max_steps:
            break

    if not x_list:
        return np.empty((0, n_feats), dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.vstack(x_list).astype(np.float32), np.array(y_list, dtype=np.float32)


def _run_sklearn_baselines(
    train_loader: DataLoader,
    val_loader: DataLoader,
    for_test_loader: DataLoader,
    itw_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    model_names: list[str] | None = None,
) -> list[dict]:
    """Train Logistic Regression and Random Forest on mel features and evaluate."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if model_names is None:
        model_names = list(SKLEARN_MODELS)

    print("\nExtracting mel features for sklearn baselines...")
    x_train, y_train = _extract_mel_features(train_loader, cfg, device, cfg.training.max_train_steps)
    x_val, y_val = _extract_mel_features(val_loader, cfg, device, cfg.training.max_val_steps)
    x_for, y_for = _extract_mel_features(for_test_loader, cfg, device)
    x_itw, y_itw = _extract_mel_features(itw_loader, cfg, device)
    print(f"  Features: train={x_train.shape}, val={x_val.shape}, for={x_for.shape}, itw={x_itw.shape}")

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_for_s = scaler.transform(x_for)
    x_itw_s = scaler.transform(x_itw)

    clfs = {
        "lr": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    results: list[dict] = []
    for name, clf in clfs.items():
        if name not in model_names:
            continue
        print(f"\nTraining {name.upper()}...")
        clf.fit(x_train_s, y_train)

        val_scores = clf.predict_proba(x_val_s)[:, 1]
        val_thr = kfold_calibrate_threshold(y_val, val_scores) if len(np.unique(y_val)) > 1 else 0.5

        for_scores = clf.predict_proba(x_for_s)[:, 1]
        itw_scores = clf.predict_proba(x_itw_s)[:, 1]

        for_metrics = compute_all_metrics(y_for, for_scores, threshold=val_thr, bootstrap=False)
        itw_metrics = compute_all_metrics(y_itw, itw_scores, threshold=val_thr, bootstrap=False)

        for_eer = for_metrics.get("EER", float("nan"))
        itw_eer = itw_metrics.get("EER", float("nan"))
        gen_gap = (
            itw_eer - for_eer
            if not np.isnan(for_eer) and not np.isnan(itw_eer)
            else float("nan")
        )

        results.append({
            "model": name,
            "params_M": 0.0,
            "for_eer": for_eer,
            "for_auc": for_metrics.get("AUC", float("nan")),
            "for_min_dcf": for_metrics.get("MinDCF", float("nan")),
            "for_f1": for_metrics.get("F1", 0.0),
            "itw_eer": itw_eer,
            "itw_auc": itw_metrics.get("AUC", float("nan")),
            "itw_min_dcf": itw_metrics.get("MinDCF", float("nan")),
            "itw_f1": itw_metrics.get("F1", 0.0),
            "gen_gap_eer": gen_gap,
            "y_true_for": y_for,
            "y_score_for": for_scores,
            "y_pred_for": (for_scores >= val_thr).astype(int),
            "y_true_itw": y_itw,
            "y_score_itw": itw_scores,
            "y_pred_itw": (itw_scores >= val_thr).astype(int),
        })
        print(f"  {name.upper()} | FoR EER={for_eer:.4f} | ITW EER={itw_eer:.4f}")

    return results


def train_model(
    model_name: str,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    itw_train_loader: DataLoader | None = None,
) -> tuple[nn.Module, float, float]:
    """Train a model from scratch and return (model, best_eer, best_threshold).

    If a checkpoint already exists at results/checkpoints/<model>/model_best.pt,
    loads it and skips training.
    """
    ckpt_path = Path(cfg.paths.checkpoint_dir) / model_name / "model_best.pt"
    model = get_model(model_name, cfg).to(device)
    print(f"  Parameters: {count_parameters(model)['trainable']:,}")

    if ckpt_path.exists():
        print(f"\nFound existing checkpoint for {model_name.upper()} — skipping training.")
        m_inner = model.module if hasattr(model, "module") else model
        m_inner.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=True))
        return model, 0.0, 0.5

    print(f"\nTraining {model_name.upper()} from scratch...")

    # RawNet2 (22M params) needs smaller batch to fit 6GB VRAM
    _SMALL_BATCH_MODELS = {"rawnet2"}
    if model_name in _SMALL_BATCH_MODELS:
        small_bs = 4
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=small_bs,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        print(f"  Reduced batch size to {small_bs} for VRAM headroom")

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

    # RawNet2: batch=4 + grad_accum=4 → effective batch=16, same quality as batch=16
    _ACCUM_MAP = {"rawnet2": 4}
    grad_accum = _ACCUM_MAP.get(model_name, 1)

    # DANN only for ConDetection; baselines train without domain adversarial loss
    use_dann_save = cfg.dann.enabled
    if model_name != "condetection":
        cfg.dann.enabled = False

    try:
        for epoch in range(1, cfg.training.epochs + 1):
            itw_iter = iter(itw_train_loader) if (itw_train_loader and cfg.dann.enabled) else None
            train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, device, cfg, epoch, itw_iter, grad_accum=grad_accum)

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

    sklearn_names = [m for m in model_names if m in SKLEARN_MODELS]
    neural_names = [m for m in model_names if m not in SKLEARN_MODELS]

    all_results: list[dict] = []
    all_models: dict[str, tuple[nn.Module, float, float]] = {}

    criterion = build_criterion(cfg, device)

    # Sklearn baselines (feature-based, no GPU training)
    if sklearn_names:
        sk_results = _run_sklearn_baselines(
            train_loader, val_loader, for_test_loader, itw_loader, cfg, device, sklearn_names
        )
        all_results.extend(sk_results)

    # ConDetection (pre-trained or train fresh)
    if "condetection" in neural_names:
        if condetection_model is not None:
            cd_model = condetection_model
            cd_threshold = condetection_threshold
        else:
            cd_model, _, cd_threshold = train_model(
                "condetection", cfg, train_loader, val_loader, device, itw_train_loader
            )
        cd_model.cpu()  # offload to CPU — free VRAM for baselines
        all_models["condetection"] = (cd_model, 0.0, cd_threshold)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Train each neural baseline
    for name in neural_names:
        if name == "condetection":
            continue
        model, best_eer, threshold = train_model(name, cfg, train_loader, val_loader, device)
        model.cpu()  # offload immediately after training
        all_models[name] = (model, best_eer, threshold)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Evaluate all models (one at a time on GPU)
    for name, (model, _, threshold) in all_models.items():
        print(f"\nEvaluating {name.upper()}...")
        model.to(device)
        for_m = evaluate(model, for_test_loader, criterion, device, cfg, threshold=threshold)
        itw_m = evaluate(model, itw_loader, criterion, device, cfg, threshold=threshold)
        params = count_parameters(model)
        model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
