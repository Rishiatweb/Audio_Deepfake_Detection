"""Light CNN (LCNN) baseline for audio deepfake detection.

Reference: Wu et al. (2020) — Light CNN for deep fake speech detection.
Operates on log-mel spectrograms (same pipeline as ConDetection for fair comparison).
Uses Max-Feature-Map (MFM) activations instead of standard activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import Config


class MFM(nn.Module):
    """Max-Feature-Map activation: splits channels in half, takes element-wise max."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2*C, H, W) → (B, C, H, W)
        x1, x2 = x.chunk(2, dim=1)
        return torch.max(x1, x2)


class LCNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.mfm = MFM(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mfm(self.bn(self.conv(x)))


class LCNN(nn.Module):
    """Light CNN operating on single-scale log-mel spectrogram (mid scale).

    Input: (B, 1, n_mels, T) from mid-resolution mel config.
    Output: logit (B,)
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        # Use mid-scale mel config (index 1)
        mid_cfg = cfg.mel_configs[1]
        self.n_mels = mid_cfg.n_mels
        self.mel_cfg_idx = 1  # which scale to use from mels_list

        self.features = nn.Sequential(
            LCNNBlock(1, 32),  # (B,32,M,T)
            nn.MaxPool2d(2, 2),  # (B,32,M/2,T/2)
            LCNNBlock(32, 48),  # (B,48,M/2,T/2)
            nn.MaxPool2d(2, 2),  # (B,48,M/4,T/4)
            LCNNBlock(48, 64),  # (B,64,M/4,T/4)
            nn.MaxPool2d(2, 2),  # (B,64,M/8,T/8)
            LCNNBlock(64, 64),  # (B,64,M/8,T/8)
            nn.MaxPool2d(2, 2),  # (B,64,M/16,T/16)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            MFM(64),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, mels_list: list[torch.Tensor]) -> tuple[torch.Tensor, list, None]:
        """Same interface as ConDetection for fair comparison.

        Returns logits, empty list (no embeddings), None (no domain).
        """
        x = mels_list[self.mel_cfg_idx]  # (B, 1, n_mels, T)
        h = self.features(x)
        h = self.global_pool(h)  # (B, 64, 1, 1)
        logits = self.classifier(h).squeeze(-1)
        return logits, [], None

    def consistency_loss(self, _embs: list) -> torch.Tensor:
        return torch.tensor(0.0)
