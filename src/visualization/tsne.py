"""t-SNE domain separation visualization for DANN analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

matplotlib.use("Agg")


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mel_configs,
    sample_rate: int = 16000,
    max_samples: int = 2000,
) -> np.ndarray:
    """Extract fused embeddings (B, D) from model's cross-scale fusion layer."""
    from src.data.spectrograms import make_multires_logmels

    model.eval()
    m_inner = model.module if hasattr(model, "module") else model

    embeddings: list[np.ndarray] = []
    n_collected = 0

    # Hook to capture fused output before classifier
    captured: list[torch.Tensor] = []

    def hook_fn(module, inp, out):
        captured.append(out[0].detach().cpu())  # out is (fused, attn_w)

    handle = m_inner.cross_scale.register_forward_hook(hook_fn)
    amp_enabled = device.type == "cuda"

    try:
        for waveforms, _ in loader:
            if n_collected >= max_samples:
                break
            waveforms = waveforms.to(device, non_blocking=True)
            mels = make_multires_logmels(waveforms, mel_configs, sample_rate, train_mode=False)
            captured.clear()
            with autocast(device_type=device.type, enabled=amp_enabled):
                model(mels)
            if captured:
                emb = captured[0].numpy()  # (B, D)
                embeddings.append(emb)
                n_collected += len(emb)
    finally:
        handle.remove()

    return np.concatenate(embeddings, axis=0)[:max_samples]


def plot_tsne_domains(
    model_with_dann: nn.Module,
    model_no_dann: nn.Module,
    for_loader: DataLoader,
    itw_loader: DataLoader,
    device: torch.device,
    mel_configs,
    sample_rate: int,
    out_path: str,
    max_samples: int = 1500,
) -> None:
    """t-SNE comparing feature spaces of DANN vs no-DANN model.

    Shows how DANN forces domain mixing while preserving class separation.
    """
    from sklearn.manifold import TSNE

    print("Extracting embeddings for t-SNE (may take 1-2 min)...")

    # Collect source and target embeddings
    for_embs_dann = extract_embeddings(model_with_dann, for_loader, device, mel_configs, sample_rate, max_samples // 2)
    itw_embs_dann = extract_embeddings(model_with_dann, itw_loader, device, mel_configs, sample_rate, max_samples // 2)
    for_embs_nodann = extract_embeddings(model_no_dann, for_loader, device, mel_configs, sample_rate, max_samples // 2)
    itw_embs_nodann = extract_embeddings(model_no_dann, itw_loader, device, mel_configs, sample_rate, max_samples // 2)

    n_for = len(for_embs_dann)
    n_itw = len(itw_embs_dann)

    all_dann = np.concatenate([for_embs_dann, itw_embs_dann], axis=0)
    all_nodann = np.concatenate([for_embs_nodann, itw_embs_nodann], axis=0)

    domain_labels = np.array([0] * n_for + [1] * n_itw)

    print("  Fitting t-SNE for DANN model...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
    z_dann = tsne.fit_transform(all_dann)

    print("  Fitting t-SNE for no-DANN model...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
    z_nodann = tsne.fit_transform(all_nodann)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = {0: "#4C72B0", 1: "#DD8452"}
    domain_names = {0: "FoR (source)", 1: "ITW (target)"}

    for ax, (z, title) in zip(axes, [(z_nodann, "Without DANN"), (z_dann, "With DANN")]):
        for d in [0, 1]:
            mask = domain_labels == d
            ax.scatter(z[mask, 0], z[mask, 1], c=palette[d], label=domain_names[d], alpha=0.5, s=12, edgecolors="none")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend()
        ax.axis("off")

    fig.suptitle("t-SNE Domain Separation: DANN Forces Domain Mixing", fontweight="bold", fontsize=14)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved t-SNE plot to {out_path}")
