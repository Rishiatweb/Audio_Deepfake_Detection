"""Training loop, evaluation, and checkpoint management."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.config import Config
from src.data.spectrograms import make_multires_logmels
from src.evaluation.metrics import compute_eer
from src.training.losses import (
    build_dann_domain_labels,
    dann_lambda_schedule,
    label_smooth,
)

EvalResult = dict  # EER, AUC, AP, F1, Acc, Prec, Rec, loss, y_true, y_score, threshold


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Grid-search threshold maximising 0.7*F1 + 0.3*balanced_accuracy."""
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return 0.5, 0.0
    best_t, best_obj = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        y_p = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_p, zero_division=0)
        rec_pos = recall_score(y_true, y_p, zero_division=0)
        rec_neg = recall_score(1 - y_true, 1 - y_p, zero_division=0)
        bal_acc = 0.5 * (rec_pos + rec_neg)
        obj = 0.7 * f1 + 0.3 * bal_acc
        if obj > best_obj:
            best_obj = float(obj)
            best_t = float(t)
    return best_t, best_obj


def kfold_calibrate_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_folds: int = 5,
) -> float:
    """K-fold threshold calibration — more robust than single-pass grid search.

    Splits (y_true, y_score) into n_folds stratified folds, finds the best
    threshold on each fold's held-out slice, returns the mean. Reduces
    overfitting to a single validation split.
    """
    from sklearn.model_selection import StratifiedKFold

    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return 0.5
    # n_folds cannot exceed smallest class count
    min_class = int(np.bincount(y_true.astype(int)).min())
    n_folds = min(n_folds, min_class)
    if n_folds < 2:
        t, _ = find_best_threshold(y_true, y_score)
        return t
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    thresholds: list[float] = []
    for _, val_idx in skf.split(y_score, y_true.astype(int)):
        t, _ = find_best_threshold(y_true[val_idx], y_score[val_idx])
        thresholds.append(t)
    return float(np.mean(thresholds))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    epoch: int,
    itw_loader_iter=None,
    grad_accum: int = 1,
) -> float:
    """Train for one epoch. Returns mean total loss."""
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_dann = 0.0
    total_disc_correct = 0
    total_disc_samples = 0
    amp_enabled = device.type == "cuda"
    tr = cfg.training
    n_batches = len(loader) if tr.max_train_steps is None else min(len(loader), tr.max_train_steps)
    steps_done = 0

    m_inner = model.module if hasattr(model, "module") else model
    dann_lam = (
        dann_lambda_schedule(epoch, tr.epochs, cfg.dann.warmup_epochs, cfg.dann.lambda_max) if cfg.dann.enabled else 0.0
    )
    if hasattr(m_inner, "_dann_lambda"):
        m_inner._dann_lambda = dann_lam

    dann_crit = nn.BCEWithLogitsLoss() if cfg.dann.enabled else None

    for step, (waveforms, labels) in enumerate(loader, 1):
        waveforms = waveforms.to(device, non_blocking=True)
        labels_gpu = label_smooth(labels.to(device, non_blocking=True), tr.label_smooth)
        src_domain = build_dann_domain_labels(waveforms.size(0), is_source=True, device=device)

        m_inner_fwd = model.module if hasattr(model, "module") else model
        is_rawnet = hasattr(m_inner_fwd, "sinc")
        mels_list = (
            [] if is_rawnet
            else make_multires_logmels(waveforms, cfg.mel_configs, cfg.audio.sample_rate, train_mode=True)
        )
        model_input = [waveforms.unsqueeze(1)] if is_rawnet else mels_list

        tgt_wavs = None
        if cfg.dann.enabled and itw_loader_iter is not None and dann_lam > 0:
            try:
                tgt_wavs, _ = next(itw_loader_iter)
            except StopIteration:
                tgt_wavs = None

        if steps_done % grad_accum == 0:
            optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            if hasattr(m_inner_fwd, "domain_disc"):
                logits, cembs, src_domain_logits = model(model_input, domain_labels=src_domain)
            else:
                out = model(model_input)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                cembs, src_domain_logits = [], None
            cls_loss = criterion(logits, labels_gpu)

            c_loss = (
                m_inner.consistency_loss(cembs)
                if (cembs and cembs[0].shape[0] > 1)
                else torch.tensor(0.0, device=device)
            )

            d_diff = getattr(m_inner, "_last_diff_loss", torch.tensor(0.0, device=device))
            if not torch.is_tensor(d_diff):
                d_diff = torch.tensor(float(d_diff), device=device)

            dann_loss = torch.tensor(0.0, device=device)
            if cfg.dann.enabled and src_domain_logits is not None and dann_lam > 0 and dann_crit is not None:
                src_domain_logits = torch.nan_to_num(src_domain_logits, nan=0.0, posinf=30.0, neginf=-30.0)
                dann_loss = dann_crit(src_domain_logits, src_domain)

                if tgt_wavs is not None:
                    tgt_wavs = tgt_wavs.to(device, non_blocking=True)
                    tgt_mels = make_multires_logmels(tgt_wavs, cfg.mel_configs, cfg.audio.sample_rate, train_mode=True)
                    tgt_domain = build_dann_domain_labels(tgt_wavs.size(0), is_source=False, device=device)
                    _, _, tgt_domain_logits = model(tgt_mels, domain_labels=tgt_domain)
                    if tgt_domain_logits is not None:
                        tgt_domain_logits = torch.nan_to_num(tgt_domain_logits, nan=0.0, posinf=30.0, neginf=-30.0)
                        dann_loss = (dann_loss + dann_crit(tgt_domain_logits, tgt_domain)) * 0.5

            if torch.is_tensor(dann_loss) and not bool(torch.isfinite(dann_loss).all()):
                dann_loss = torch.zeros((), device=device)

            loss = (cls_loss + tr.lambda_c * c_loss + cfg.diffusion.lambda_diff * d_diff + dann_loss) / grad_accum

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        if (steps_done + 1) % grad_accum == 0 or (steps_done + 1) == n_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        total_loss += loss.detach().float().item()
        total_cls += cls_loss.detach().float().item()
        total_dann += dann_loss.detach().float().item() if torch.is_tensor(dann_loss) else float(dann_loss)
        steps_done += 1

        # Track domain discriminator accuracy (target: ~0.5 = confused)
        if cfg.dann.enabled and src_domain_logits is not None and dann_lam > 0:
            with torch.no_grad():
                src_pred = (src_domain_logits.detach().sigmoid() > 0.5).float()
                src_gt = (src_domain > 0.5).float()
                total_disc_correct += (src_pred == src_gt).sum().item()
                total_disc_samples += src_domain_logits.size(0)

        if step % 50 == 0 or step == n_batches:
            disc_acc_str = ""
            if total_disc_samples > 0:
                disc_acc = total_disc_correct / total_disc_samples
                disc_acc_str = f" | disc_acc {disc_acc:.3f}"
            print(
                f"  step {step}/{n_batches} | "
                f"cls {total_cls / steps_done:.4f} | "
                f"dann {total_dann / steps_done:.4f} | "
                f"dann_lam {dann_lam:.3f}"
                f"{disc_acc_str} | "
                f"total {total_loss / steps_done:.4f}"
            )

        if step % 25 == 0:
            del mels_list, waveforms, logits, cembs, cls_loss, c_loss, d_diff, loss, dann_loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if tr.max_train_steps is not None and step >= tr.max_train_steps:
            break

    return total_loss / max(steps_done, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    threshold: float = 0.5,
    max_steps: int | None = None,
) -> EvalResult:
    """Evaluate model on a DataLoader. Returns dict with all metrics."""
    model.eval()
    total_loss = 0.0
    all_scores: list[float] = []
    all_labels: list[float] = []
    amp_enabled = device.type == "cuda"
    steps_done = 0

    for step, (waveforms, labels) in enumerate(loader, 1):
        waveforms = waveforms.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)
        mels_list = make_multires_logmels(waveforms, cfg.mel_configs, cfg.audio.sample_rate, train_mode=False)
        _m = model.module if hasattr(model, "module") else model
        model_input = [waveforms.unsqueeze(1)] if hasattr(_m, "sinc") else mels_list

        with autocast(device_type=device.type, enabled=amp_enabled):
            out = model(model_input)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss = criterion(logits, label_smooth(labels_d, cfg.training.label_smooth))

        total_loss += loss.detach().float().item()
        all_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
        steps_done += 1

        if step % 50 == 0:
            del mels_list, waveforms, labels_d, logits, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if max_steps is not None and step >= max_steps:
            break

    y_t = np.array(all_labels)
    y_s = np.array(all_scores)
    finite = np.isfinite(y_t) & np.isfinite(y_s)
    y_t, y_s = y_t[finite], y_s[finite]

    if len(y_t) == 0 or np.unique(y_t).size < 2:
        return dict(
            loss=total_loss / max(steps_done, 1),
            EER=np.nan,
            AUC=np.nan,
            AP=np.nan,
            F1=0.0,
            Acc=0.0,
            Prec=0.0,
            Rec=0.0,
            y_true=y_t,
            y_score=y_s,
            threshold=float(threshold),
        )

    y_p = (y_s >= threshold).astype(int)
    return dict(
        loss=total_loss / max(steps_done, 1),
        EER=compute_eer(y_t, y_s),
        AUC=roc_auc_score(y_t, y_s),
        AP=average_precision_score(y_t, y_s),
        F1=f1_score(y_t, y_p, zero_division=0),
        Acc=accuracy_score(y_t, y_p),
        Prec=precision_score(y_t, y_p, zero_division=0),
        Rec=recall_score(y_t, y_p, zero_division=0),
        y_true=y_t,
        y_score=y_s,
        threshold=float(threshold),
    )


@torch.no_grad()
def evaluate_tta(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    threshold: float = 0.5,
    tta_shifts: tuple[int, ...] = (0,),
) -> EvalResult:
    """Test-time augmentation: average scores across waveform shifts."""
    model.eval()
    total_loss = 0.0
    all_scores: list[float] = []
    all_labels: list[float] = []
    amp_enabled = device.type == "cuda"
    steps_done = 0

    for step, (waveforms, labels) in enumerate(loader, 1):
        waveforms = waveforms.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)

        _m_tta = model.module if hasattr(model, "module") else model
        score_sum = None
        first_logits = None
        for sh in tta_shifts:
            wf = waveforms if int(sh) == 0 else torch.roll(waveforms, shifts=int(sh), dims=1)
            mels_list = make_multires_logmels(wf, cfg.mel_configs, cfg.audio.sample_rate, train_mode=False)
            model_input = [wf.unsqueeze(1)] if hasattr(_m_tta, "sinc") else mels_list
            with autocast(device_type=device.type, enabled=amp_enabled):
                out = model(model_input)
                logits = out[0] if isinstance(out, (tuple, list)) else out
            if first_logits is None:
                first_logits = logits
            probs = torch.sigmoid(logits)
            score_sum = probs if score_sum is None else (score_sum + probs)

        avg_scores = score_sum / max(len(tta_shifts), 1)
        with autocast(device_type=device.type, enabled=amp_enabled):
            loss = criterion(first_logits, label_smooth(labels_d, cfg.training.label_smooth))

        total_loss += loss.detach().float().item()
        all_scores.extend(avg_scores.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
        steps_done += 1

        if (max_steps := cfg.training.max_val_steps) and step >= max_steps:
            break

    y_t = np.array(all_labels)
    y_s = np.array(all_scores)
    finite = np.isfinite(y_t) & np.isfinite(y_s)
    y_t, y_s = y_t[finite], y_s[finite]

    if len(y_t) == 0 or np.unique(y_t).size < 2:
        return dict(
            loss=total_loss / max(steps_done, 1),
            EER=np.nan,
            AUC=np.nan,
            AP=np.nan,
            F1=0.0,
            Acc=0.0,
            Prec=0.0,
            Rec=0.0,
            y_true=y_t,
            y_score=y_s,
            threshold=float(threshold),
        )

    y_p = (y_s >= threshold).astype(int)
    return dict(
        loss=total_loss / max(steps_done, 1),
        EER=compute_eer(y_t, y_s),
        AUC=roc_auc_score(y_t, y_s),
        AP=average_precision_score(y_t, y_s),
        F1=f1_score(y_t, y_p, zero_division=0),
        Acc=accuracy_score(y_t, y_p),
        Prec=precision_score(y_t, y_p, zero_division=0),
        Rec=recall_score(y_t, y_p, zero_division=0),
        y_true=y_t,
        y_score=y_s,
        threshold=float(threshold),
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_val_eer: float,
    best_threshold: float,
    ckpt_dir: str,
    tag: str = "best",
) -> None:
    """Save full training checkpoint."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    m = model.module if hasattr(model, "module") else model
    state = m.state_dict()
    torch.save(state, ckpt_dir / f"model_{tag}.pt")
    torch.save(
        {
            "epoch": epoch,
            "best_val_eer": best_val_eer,
            "best_threshold": best_threshold,
            "model_state_dict": state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        },
        ckpt_dir / f"checkpoint_{tag}.pth",
    )
    (ckpt_dir / "best_threshold.txt").write_text(f"{best_threshold:.6f}\n", encoding="utf-8")
