"""Multi-resolution log-mel spectrogram extraction (CPU and batched GPU)."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.config import MelConfig

# Module-level caches for mel filterbanks and windows (keyed by config)
_MEL_FB_CACHE: dict = {}
_WIN_CACHE: dict = {}


def compute_logmel_cpu(y: np.ndarray, cfg: MelConfig, sample_rate: int = 16000) -> np.ndarray:
    """Single-sample log-mel spectrogram on CPU via librosa."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    mel = np.maximum(mel, 1e-10)
    lm = librosa.power_to_db(mel, ref=np.max)
    lm = np.nan_to_num(lm, nan=0.0, posinf=0.0, neginf=-80.0)

    std = lm.std()
    if std < 1e-6:
        return lm.astype(np.float32)
    return ((lm - lm.mean()) / (std + 1e-6)).astype(np.float32)


def _get_window(n_fft: int, device: torch.device) -> torch.Tensor:
    key = (n_fft, str(device))
    if key not in _WIN_CACHE:
        _WIN_CACHE[key] = torch.hann_window(n_fft, device=device)
    return _WIN_CACHE[key]


def _get_mel_filter(cfg: MelConfig, device: torch.device, sample_rate: int = 16000) -> torch.Tensor:
    import librosa

    key = (cfg.n_fft, cfg.n_mels, sample_rate, str(device))
    if key not in _MEL_FB_CACHE:
        fb = librosa.filters.mel(sr=sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels).astype(np.float32)
        _MEL_FB_CACHE[key] = torch.from_numpy(fb).to(device)
    return _MEL_FB_CACHE[key]


def _apply_light_specaugment(logmel: torch.Tensor) -> torch.Tensor:
    """In-place specaugment on a batch of log-mel tensors (B, M, T)."""
    B, M, T = logmel.shape
    max_f = max(2, M // 12)
    max_t = max(2, T // 16)
    for b in range(B):
        if random.random() < 0.5:
            f = random.randint(1, max_f)
            f0 = random.randint(0, max(0, M - f))
            logmel[b, f0 : f0 + f, :] = 0.0
        if random.random() < 0.5:
            t = random.randint(1, max_t)
            t0 = random.randint(0, max(0, T - t))
            logmel[b, :, t0 : t0 + t] = 0.0
    return logmel


def make_multires_logmels(
    waveforms: torch.Tensor,
    mel_configs: list[MelConfig],
    sample_rate: int = 16000,
    train_mode: bool = False,
) -> list[torch.Tensor]:
    """Batched GPU log-mel extraction for all scales.

    Args:
        waveforms: (B, num_samples) float32 in [-1, 1]
        mel_configs: list of MelConfig (one per scale)
        sample_rate: audio sample rate
        train_mode: if True, apply light SpecAugment

    Returns:
        List of K tensors each of shape (B, 1, n_mels, T)
    """
    waveforms = torch.nan_to_num(waveforms.float(), nan=0.0, posinf=0.0, neginf=0.0)
    device = waveforms.device
    num_samples = waveforms.shape[-1]
    mels_out = []

    for cfg in mel_configs:
        t_target = math.ceil(num_samples / cfg.hop_length) + 1

        spec = torch.stft(
            waveforms,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.n_fft,
            window=_get_window(cfg.n_fft, device),
            center=True,
            return_complex=True,
        )
        power = spec.real.pow(2) + spec.imag.pow(2)
        mel_fb = _get_mel_filter(cfg, device, sample_rate)
        mel = torch.einsum("mf,bft->bmt", mel_fb, power)

        logmel = torch.log(mel + 1e-6)
        mean = logmel.mean(dim=(1, 2), keepdim=True)
        std = logmel.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        logmel = torch.nan_to_num((logmel - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

        if train_mode:
            logmel = _apply_light_specaugment(logmel)

        if logmel.shape[-1] < t_target:
            logmel = F.pad(logmel, (0, t_target - logmel.shape[-1]))
        else:
            logmel = logmel[..., :t_target]

        mels_out.append(logmel.unsqueeze(1).contiguous())

    return mels_out
