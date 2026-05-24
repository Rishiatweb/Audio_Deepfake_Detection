"""RawNet2 baseline for audio deepfake detection.

Reference: Tak et al. (2021) — End-to-End anti-spoofing with RawNet2.

RawNet2 operates directly on raw waveforms (no spectrogram).
Uses SincConv front-end + ResBlocks + GRU.

This implementation is trained from scratch on FoR for fair comparison.
Input: raw waveform (B, num_samples). Extracted from mels_list[0] which
stores the waveform when model_type == "rawnet2" (see factory.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config


class SincConv(nn.Module):
    """SincConv layer: learnable bandpass filters on raw audio.

    Reference: Ravanelli & Bengio (2018) — SincNet.
    """

    def __init__(self, out_channels: int, kernel_size: int = 1024, sample_rate: int = 16000) -> None:
        super().__init__()
        assert kernel_size % 2 == 0, "kernel_size must be even"
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Initialise filter cutoff frequencies
        low_hz = 30.0
        high_hz = sample_rate / 2 - 100.0
        mel_low = 2595 * math.log10(1 + low_hz / 700)
        mel_high = 2595 * math.log10(1 + high_hz / 700)
        mel_points = torch.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        self.low_hz_ = nn.Parameter(hz_points[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz_points[1:] - hz_points[:-1]).unsqueeze(1))

        n = (kernel_size - 1) / 2.0
        t = torch.arange(-n, n + 1).float() / sample_rate
        self.register_buffer("n_", t.view(1, -1))
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(kernel_size).float() / kernel_size)
        self.register_buffer("window_", window.view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = torch.clamp(self.low_hz_, min=50.0) / self.sample_rate
        high = torch.clamp(low + torch.clamp(self.band_hz_, min=10.0) / self.sample_rate, max=0.5 - 1e-6)

        f_times_t = self.n_.expand(self.out_channels, -1)  # type: ignore[attr-defined]
        low_pass1 = 2 * low * torch.sinc(2 * low * f_times_t)
        low_pass2 = 2 * high * torch.sinc(2 * high * f_times_t)
        band_pass = low_pass2 - low_pass1

        band_pass = band_pass * self.window_  # type: ignore[attr-defined]
        band_pass = band_pass / (band_pass.abs().max(dim=1, keepdim=True)[0] + 1e-6)

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, padding=self.kernel_size // 2, stride=1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )
        self.fms = nn.Linear(out_ch, out_ch)  # Feature Map Scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        # FMS: channel-wise scaling from global avg
        scale = torch.sigmoid(self.fms(h.mean(dim=-1))).unsqueeze(-1)
        return F.silu(h * scale + self.skip(x))


class RawNet2(nn.Module):
    """RawNet2: SincConv + ResBlocks + GRU classifier.

    Input: raw waveform (B, num_samples) extracted from dataset.
    Interface: forward(mels_list) where mels_list[0] is the raw waveform
    wrapped as (B, 1, num_samples) to maintain consistency.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        sr = cfg.audio.sample_rate
        self.sinc = SincConv(out_channels=70, kernel_size=1024, sample_rate=sr)
        self.bn_sinc = nn.BatchNorm1d(70)

        self.res_blocks = nn.Sequential(
            ResBlock1D(70, 128),
            ResBlock1D(128, 128),
            ResBlock1D(128, 256, stride=2),
            ResBlock1D(256, 256),
            ResBlock1D(256, 512, stride=2),
            ResBlock1D(512, 512),
        )

        self.gru = nn.GRU(input_size=512, hidden_size=1024, num_layers=3, batch_first=True, dropout=0.3)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, mels_list: list[torch.Tensor]) -> tuple[torch.Tensor, list, None]:
        """mels_list[0] contains raw waveform (B, 1, num_samples) for RawNet2."""
        # Extract raw waveform from the first item
        wav = mels_list[0].squeeze(1)  # (B, num_samples)

        x = self.sinc(wav.unsqueeze(1))  # (B, 70, T)
        x = F.silu(self.bn_sinc(x))
        x = self.res_blocks(x)  # (B, 512, T')
        x = x.transpose(1, 2)  # (B, T', 512)

        _, h_n = self.gru(x)  # h_n: (3, B, 1024)
        h = h_n[-1]  # (B, 1024) last layer hidden state

        logits = self.classifier(h).squeeze(-1)
        return logits, [], None

    def consistency_loss(self, _embs: list) -> torch.Tensor:
        return torch.tensor(0.0)
