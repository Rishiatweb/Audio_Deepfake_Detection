"""Grad-CAM saliency maps for ConDetection-DANN."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")


def compute_gradcam(
    model: nn.Module,
    mels_list: list[torch.Tensor],
    device: torch.device,
) -> list[np.ndarray]:
    """Compute Grad-CAM saliency maps for each spectral scale.

    Registers hooks on scale encoders, runs one forward+backward pass,
    and returns one normalised heatmap per scale.

    Args:
        model: ConDetection (or wrapped) in eval mode
        mels_list: list of K tensors (1, 1, n_mels, T)
        device: torch device

    Returns:
        list of K numpy arrays shape (n_mels, T) — saliency per scale
    """
    m_inner = model.module if hasattr(model, "module") else model

    activations: dict[int, torch.Tensor] = {}
    gradients: dict[int, torch.Tensor] = {}
    handles = []

    # Register hooks on each scale encoder's projection (last linear layer)
    for k, encoder in enumerate(m_inner.scale_encoders):

        def fwd_hook(module, inp, out, k=k):
            activations[k] = out  # (1, T, D)

        def bwd_hook(module, grad_in, grad_out, k=k):
            gradients[k] = grad_out[0]  # (1, T, D)

        handles.append(encoder.proj.register_forward_hook(fwd_hook))
        handles.append(encoder.proj.register_full_backward_hook(bwd_hook))

    model.eval()
    mels = [m.to(device) for m in mels_list]

    try:
        # Forward + backward
        logits, _, _ = model(mels)
        model.zero_grad()
        logits.sum().backward()

        cams: list[np.ndarray] = []
        for k, mel in enumerate(mels):
            if k not in activations or k not in gradients:
                cams.append(np.zeros((mel.shape[2], mel.shape[3])))
                continue

            act = activations[k].detach()  # (1, T, D)
            grad = gradients[k].detach()  # (1, T, D)

            weights = grad.mean(dim=1)  # (1, D)
            cam = (weights.unsqueeze(1) * act).sum(dim=-1)  # (1, T)
            cam = cam.squeeze(0).cpu().numpy()  # (T,)

            cam = np.maximum(cam, 0)
            if cam.max() > 1e-8:
                cam = cam / cam.max()

            # Resize to (n_mels, T) for overlay
            n_mels = mel.shape[2]
            cam_2d = np.tile(cam[np.newaxis, :], (n_mels, 1))
            cams.append(cam_2d)
    finally:
        for h in handles:
            h.remove()
        model.zero_grad()

    return cams


def plot_gradcam(
    mels_list: list[torch.Tensor],
    cams: list[np.ndarray],
    mel_names: list[str],
    out_path: str,
    title: str = "Grad-CAM Saliency Maps",
    n_samples: int = 3,
) -> None:
    """Plot spectrograms with Grad-CAM overlays for multiple samples.

    mels_list: list of K tensors (B, 1, n_mels, T)
    cams: list of K arrays (n_mels, T)
    """
    K = len(mels_list)
    n_samples = min(n_samples, mels_list[0].shape[0])
    fig, axes = plt.subplots(n_samples, K * 2, figsize=(4 * K * 2, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for s in range(n_samples):
        for k in range(K):
            mel = mels_list[k][s, 0].cpu().numpy()  # (n_mels, T)
            cam = cams[k]

            col_mel = k * 2
            col_cam = k * 2 + 1

            axes[s, col_mel].imshow(mel, aspect="auto", origin="lower", cmap="magma")
            axes[s, col_mel].set_title(f"Scale: {mel_names[k]}")
            axes[s, col_mel].axis("off")

            axes[s, col_cam].imshow(mel, aspect="auto", origin="lower", cmap="magma", alpha=0.6)
            axes[s, col_cam].imshow(cam, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
            axes[s, col_cam].set_title(f"Grad-CAM: {mel_names[k]}")
            axes[s, col_cam].axis("off")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
