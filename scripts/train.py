"""Main training entrypoint for ConDetection-DANN."""
from __future__ import annotations

import argparse
import gc
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import load_config
from src.data.datasets import FastAudioDataset, build_splits, make_loaders
from src.data.spectrograms import make_multires_logmels
from src.models.factory import count_parameters, get_model
from src.training.losses import build_criterion, dann_lambda_schedule
from src.training.scheduler import get_cosine_schedule_with_warmup
from src.training.trainer import (
    evaluate,
    evaluate_tta,
    kfold_calibrate_threshold,
    save_checkpoint,
    train_one_epoch,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train ConDetection-DANN")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    p.add_argument("--model", default="condetection", help="Model name (condetection/aasist/lcnn/rawnet2)")
    p.add_argument("--for-base", default=None, help="Override FoR dataset root")
    p.add_argument("--itw-root", default=None, help="Override ITW dataset root")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    p.add_argument("--no-dann", action="store_true", help="Disable DANN")
    return p.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.for_base:
        cfg.paths.for_base = args.for_base
    if args.itw_root:
        cfg.paths.itw_root = args.itw_root
    if args.output_dir:
        cfg.paths.output_dir = args.output_dir
        cfg.paths.checkpoint_dir = f"{args.output_dir}/checkpoints"
        cfg.paths.figures_dir = f"{args.output_dir}/figures"
    if args.no_dann:
        cfg.dann.enabled = False

    set_seeds(cfg.training.seed)
    cfg.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # ─── Data ───
    print("Loading datasets...")
    train_df, val_df, for_test_df, itw_df = build_splits(cfg)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  FoR test: {len(for_test_df):,}  ITW: {len(itw_df):,}")

    train_loader, val_loader, for_test_loader, itw_loader = make_loaders(train_df, val_df, for_test_df, itw_df, cfg)

    itw_train_loader = None
    if cfg.dann.enabled and len(itw_df) > 0:
        itw_sample = itw_df.sample(min(cfg.training.max_train_samples, len(itw_df)), random_state=cfg.training.seed).reset_index(drop=True)
        itw_train_ds = FastAudioDataset(itw_sample, augment=True, cfg=cfg)
        itw_train_loader = torch.utils.data.DataLoader(
            itw_train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0, drop_last=True
        )

    # ─── Model ───
    model = get_model(args.model, cfg).to(device)
    params = count_parameters(model)
    print(f"Model: {args.model} | trainable params: {params['trainable']:,}")

    n_real = int((train_df.label == 0).sum())
    n_fake = int((train_df.label == 1).sum())
    criterion = build_criterion(cfg, device, pos_weight_val=n_real / max(n_fake, 1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=1e-2)
    effective_steps = len(train_loader) if cfg.training.max_train_steps is None else cfg.training.max_train_steps
    total_steps = cfg.training.epochs * effective_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    # ─── Training loop ───
    history = []
    best_eer = 1.0
    best_threshold = 0.5
    patience_cnt = 0
    t0 = perf_counter()

    for epoch in range(1, cfg.training.epochs + 1):
        ep_t0 = perf_counter()
        itw_iter = iter(itw_train_loader) if itw_train_loader else None

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, device, cfg, epoch, itw_iter,
        )

        val_raw = evaluate(model, val_loader, criterion, device, cfg, threshold=0.5, max_steps=cfg.training.max_val_steps)
        auc = val_raw.get("AUC")
        val_thr = 0.5
        if auc is not None and auc == auc:
            val_thr = kfold_calibrate_threshold(val_raw["y_true"], val_raw["y_score"])

        val_m = evaluate(model, val_loader, criterion, device, cfg, threshold=val_thr, max_steps=cfg.training.max_val_steps)
        eer = val_m["EER"]
        flag = ""

        if eer == eer and eer < best_eer:  # not nan and improving
            best_eer = eer
            best_threshold = val_thr
            patience_cnt = 0
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_eer, best_threshold, cfg.paths.checkpoint_dir)
            flag = "  * best"
        else:
            patience_cnt += 1

        dann_lam = dann_lambda_schedule(epoch, cfg.training.epochs, cfg.dann.warmup_epochs, cfg.dann.lambda_max)
        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_m["loss"],
                   EER=eer, AUC=val_m["AUC"], F1=val_m["F1"], Acc=val_m["Acc"],
                   threshold=val_thr, dann_lambda=dann_lam, time_s=perf_counter() - ep_t0)
        history.append(row)

        print(
            f"Ep {epoch:02d}/{cfg.training.epochs} | "
            f"loss {train_loss:.4f}→{val_m['loss']:.4f} | "
            f"EER {eer:.4f} | AUC {val_m['AUC']:.4f} | "
            f"F1 {val_m['F1']:.4f} | thr {val_thr:.3f} | "
            f"DANN_lam {dann_lam:.3f}{flag}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if patience_cnt >= cfg.training.patience:
            print("Early stopping.")
            break

    total_t = perf_counter() - t0
    print(f"\nTraining complete in {total_t/60:.1f} min | Best val EER: {best_eer:.4f}")

    # Save history
    pd.DataFrame(history).to_csv(f"{cfg.paths.output_dir}/training_history.csv", index=False)

    # ─── Final evaluation ───
    ckpt = Path(cfg.paths.checkpoint_dir) / "model_best.pt"
    if ckpt.exists():
        m_inner = model.module if hasattr(model, "module") else model
        m_inner.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
        print(f"Loaded best checkpoint from {ckpt}")

    print("\nFinal evaluation:")
    eval_fn = evaluate_tta if cfg.training.use_tta else evaluate
    tta_kwargs = {"tta_shifts": tuple(cfg.training.tta_shifts)} if cfg.training.use_tta else {}

    for_m = eval_fn(model, for_test_loader, criterion, device, cfg, threshold=best_threshold, **tta_kwargs)
    itw_m = eval_fn(model, itw_loader, criterion, device, cfg, threshold=best_threshold, **tta_kwargs)

    print(f"  FoR Test  | EER={for_m['EER']:.4f} | AUC={for_m['AUC']:.4f} | F1={for_m['F1']:.4f}")
    print(f"  ITW       | EER={itw_m['EER']:.4f} | AUC={itw_m['AUC']:.4f} | F1={itw_m['F1']:.4f}")
    if for_m["EER"] == for_m["EER"] and itw_m["EER"] == itw_m["EER"]:
        print(f"  Gen Gap   | ΔEER={itw_m['EER'] - for_m['EER']:.4f}")

    # ─── DANN domain discriminator accuracy analysis ───
    if cfg.dann.enabled and hasattr(model.module if hasattr(model, "module") else model, "domain_disc"):
        print("\nDANN Domain Discriminator Accuracy Analysis...")
        m_inner = model.module if hasattr(model, "module") else model
        m_inner._dann_lambda = 0.0  # Disable GRL at eval — measure raw discriminator
        model.eval()

        @torch.no_grad()
        def collect_domain_preds(loader, domain_lbl, n_max=400):
            preds, labels = [], []
            count = 0
            for wavs, _ in loader:
                wavs = wavs.to(device)
                mels = make_multires_logmels(wavs, cfg.mel_configs, cfg.audio.sample_rate, train_mode=False)
                embeddings = []
                for mel, enc in zip(mels, m_inner.scale_encoders):
                    x = enc(mel)
                    for i, blk in enumerate(m_inner.conformer):
                        x = blk(x)
                        if i in m_inner.pool_positions:
                            x = m_inner._pool_time(x)
                    embeddings.append(x.mean(dim=1))
                fused, _ = m_inner.cross_scale(embeddings)
                dom_logit = m_inner.domain_disc.net(fused).squeeze(-1)
                dom_pred = (torch.sigmoid(dom_logit) >= 0.5).cpu().numpy().astype(int)
                preds.extend(dom_pred.tolist())
                labels.extend([domain_lbl] * len(dom_pred))
                count += len(dom_pred)
                if count >= n_max:
                    break
            return preds, labels

        for_preds, for_lbls = collect_domain_preds(for_test_loader, 0)
        itw_preds, itw_lbls = collect_domain_preds(itw_loader, 1)
        all_preds = np.array(for_preds + itw_preds)
        all_lbls = np.array(for_lbls + itw_lbls)
        from sklearn.metrics import accuracy_score as _acc
        dom_acc = _acc(all_lbls, all_preds)
        # Distance from chance (0.5) is the true measure — closer to 0 = better
        dom_confusion = abs(dom_acc - 0.5)
        print(f"\nDomain discriminator accuracy: {dom_acc:.4f}")
        print(f"  Distance from chance (0.5):  {dom_confusion:.4f}  (lower = more domain-invariant)")
        print("  0.50 = fully confused = DANN working perfectly (ideal)")
        print("  1.00 or 0.00 = systematic bias = DANN not working")
        if dom_confusion < 0.10:
            print("  EXCELLENT: Near-chance discrimination — domain-invariant features achieved")
        elif dom_confusion < 0.25:
            print("  GOOD: Partial domain invariance")
        else:
            print("  WEAK: Consider increasing lambda_max or more epochs")


if __name__ == "__main__":
    main()
