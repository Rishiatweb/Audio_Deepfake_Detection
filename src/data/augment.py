"""Audio and spectrogram augmentations for domain generalization."""

from __future__ import annotations

import random

import numpy as np


def spec_augment(
    logmel: np.ndarray,
    freq_mask_param: int = 15,
    time_mask_param: int = 30,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
) -> np.ndarray:
    """SpecAugment: mask random frequency and time bands (Park et al. 2019)."""
    mel = logmel.copy()
    n_mels, T = mel.shape
    for _ in range(n_freq_masks):
        f = random.randint(1, max(2, freq_mask_param))
        f0 = random.randint(0, max(0, n_mels - f))
        mel[f0 : f0 + f, :] = 0.0
    for _ in range(n_time_masks):
        t = random.randint(1, max(2, time_mask_param))
        t0 = random.randint(0, max(0, T - t))
        mel[:, t0 : t0 + t] = 0.0
    return mel


def additive_noise(y: np.ndarray, snr_db_range: tuple[float, float] = (12.0, 35.0)) -> np.ndarray:
    """Add Gaussian noise at a random SNR within the given range."""
    snr_db = random.uniform(*snr_db_range)
    signal_power = float(np.mean(y**2)) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*y.shape).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(y + noise, -1.0, 1.0)


def time_stretch_aug(y: np.ndarray, num_samples: int, rate_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Pitch-preserving time-stretch via librosa."""
    import librosa  # lazy import — not always needed

    rate = random.uniform(*rate_range)
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    if len(y_stretched) > num_samples:
        y_stretched = y_stretched[:num_samples]
    elif len(y_stretched) < num_samples:
        y_stretched = np.pad(y_stretched, (0, num_samples - len(y_stretched)))
    return y_stretched.astype(np.float32)


def random_channel_aug(y: np.ndarray) -> np.ndarray:
    """Cheap channel/codec-style simulation: pre/de-emphasis, LPF, clipping, mild reverb."""
    out = y.astype(np.float32)

    if random.random() < 0.35:
        a = random.uniform(0.90, 0.98)
        out[1:] = out[1:] - a * out[:-1]

    if random.random() < 0.35:
        k = random.choice([3, 5, 7])
        ker = np.ones(k, dtype=np.float32) / k
        out = np.convolve(out, ker, mode="same").astype(np.float32)

    if random.random() < 0.25:
        clip_v = random.uniform(0.65, 0.90)
        out = np.clip(out, -clip_v, clip_v) / clip_v

    if random.random() < 0.20:
        L = random.randint(24, 64)
        ir = np.exp(-np.linspace(0, 3.5, L)).astype(np.float32)
        ir[:: max(2, L // 8)] *= random.uniform(0.85, 1.15)
        ir = ir / (np.sum(np.abs(ir)) + 1e-6)
        out = np.convolve(out, ir, mode="same").astype(np.float32)

    return np.clip(out, -1.0, 1.0)


def apply_waveform_augmentations(y: np.ndarray, label: float, num_samples: int) -> np.ndarray:
    """Apply all waveform augmentations. Slightly stronger for fake class."""
    p_boost = 1.20 if label > 0.5 else 1.00

    if random.random() < min(0.45 * p_boost, 0.85):
        y = additive_noise(y)
    if random.random() < min(0.30 * p_boost, 0.70):
        y = random_channel_aug(y)
    if random.random() < 0.20:
        y = time_stretch_aug(y, num_samples)
    if random.random() < 0.30:
        gain = random.uniform(0.6, 1.4)
        y = np.clip(y * gain, -1.0, 1.0)
    if random.random() < 0.25:
        sr_eighth = num_samples // 32  # small shift
        shift = random.randint(-sr_eighth, sr_eighth)
        y = np.roll(y, shift)
    if random.random() < 0.10:
        y = -y

    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
