"""Tests for data loading, augmentation, and spectrogram extraction."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.augment import (
    additive_noise,
    apply_waveform_augmentations,
    random_channel_aug,
    spec_augment,
    time_stretch_aug,
)
from src.data.datasets import FastAudioDataset, load_audio, stratified_cap
from src.data.spectrograms import make_multires_logmels

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


@pytest.fixture
def dummy_wav(tmp_path: Path) -> Path:
    """Create a short WAV file for testing."""
    sr = 16000
    duration = 4
    n = sr * duration
    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        data = (np.sin(2 * np.pi * 440 * np.arange(n) / sr) * 32767).astype(np.int16)
        f.writeframes(data.tobytes())
    return wav_path


@pytest.fixture
def dummy_audio() -> np.ndarray:
    n = 64000
    t = np.arange(n) / 16000.0
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


# ─── load_audio ──────────────────────────────────────────────────────────────


def test_load_audio_correct_length(dummy_wav):
    y = load_audio(str(dummy_wav), sample_rate=16000, num_samples=64000)
    assert len(y) == 64000


def test_load_audio_peak_normalised(dummy_wav):
    y = load_audio(str(dummy_wav), num_samples=64000)
    assert np.max(np.abs(y)) <= 1.0 + 1e-6


def test_load_audio_float32(dummy_wav):
    y = load_audio(str(dummy_wav), num_samples=64000)
    assert y.dtype == np.float32


def test_load_audio_missing_file_returns_zeros():
    y = load_audio("/nonexistent/path/audio.wav", num_samples=64000)
    assert y.shape == (64000,)
    assert np.all(y == 0)


def test_load_audio_short_file_zero_padded(tmp_path):
    sr = 16000
    short_n = 8000  # 0.5s
    wav_path = tmp_path / "short.wav"
    with wave.open(str(wav_path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(np.zeros(short_n, dtype=np.int16).tobytes())
    y = load_audio(str(wav_path), num_samples=64000)
    assert len(y) == 64000


# ─── Augmentations ───────────────────────────────────────────────────────────


def test_additive_noise_preserves_length(dummy_audio):
    out = additive_noise(dummy_audio.copy())
    assert len(out) == len(dummy_audio)


def test_additive_noise_changes_signal(dummy_audio):
    out = additive_noise(dummy_audio.copy(), snr_db_range=(0.0, 1.0))
    assert not np.allclose(out, dummy_audio)


def test_time_stretch_preserves_length(dummy_audio):
    out = time_stretch_aug(dummy_audio.copy(), num_samples=64000)
    assert len(out) == 64000


def test_random_channel_aug_preserves_length(dummy_audio):
    out = random_channel_aug(dummy_audio.copy())
    assert len(out) == len(dummy_audio)


def test_spec_augment_preserves_shape():
    logmel = np.random.randn(80, 251).astype(np.float32)
    out = spec_augment(logmel)
    assert out.shape == logmel.shape


def test_spec_augment_zeros_some_bands():
    logmel = np.ones((80, 251), dtype=np.float32)
    out = spec_augment(logmel, n_freq_masks=2, n_time_masks=2)
    assert (out == 0).any()


def test_apply_waveform_augmentations_preserves_length(dummy_audio):
    out = apply_waveform_augmentations(dummy_audio.copy(), label=1.0, num_samples=64000)
    assert len(out) == 64000
    assert out.dtype == np.float32


# ─── FastAudioDataset ────────────────────────────────────────────────────────


def test_dataset_len_and_getitem(tmp_path, cfg):
    import pandas as pd

    # Create two dummy wav files
    sr = 16000
    for _i, name in enumerate(["real.wav", "fake.wav"]):
        with wave.open(str(tmp_path / name), "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(np.zeros(sr * 4, dtype=np.int16).tobytes())

    df = pd.DataFrame(
        [
            {"filepath": str(tmp_path / "real.wav"), "label": 0},
            {"filepath": str(tmp_path / "fake.wav"), "label": 1},
        ]
    )
    ds = FastAudioDataset(df, augment=False, cfg=cfg)

    assert len(ds) == 2
    wav, lbl = ds[0]
    assert isinstance(wav, torch.Tensor)
    assert wav.shape == (64000,)
    assert lbl.dtype == torch.float32


# ─── stratified_cap ──────────────────────────────────────────────────────────


def test_stratified_cap_reduces_size():
    import pandas as pd

    df = pd.DataFrame({"label": [0] * 100 + [1] * 100, "filepath": ["x"] * 200})
    out = stratified_cap(df, n=50)
    assert len(out) == 50


def test_stratified_cap_preserves_ratio():
    import pandas as pd

    df = pd.DataFrame({"label": [0] * 100 + [1] * 100, "filepath": ["x"] * 200})
    out = stratified_cap(df, n=50)
    ratio = (out["label"] == 0).sum() / len(out)
    assert abs(ratio - 0.5) < 0.15


# ─── make_multires_logmels ───────────────────────────────────────────────────


def test_multires_logmels_output_count(cfg):
    waveforms = torch.randn(2, cfg.audio.num_samples)
    mels = make_multires_logmels(waveforms, cfg.mel_configs, cfg.audio.sample_rate)
    assert len(mels) == len(cfg.mel_configs)


def test_multires_logmels_shapes(cfg):
    B = 2
    waveforms = torch.randn(B, cfg.audio.num_samples)
    mels = make_multires_logmels(waveforms, cfg.mel_configs, cfg.audio.sample_rate)
    for mel, mc in zip(mels, cfg.mel_configs):
        assert mel.shape[0] == B
        assert mel.shape[1] == 1  # channel dim
        assert mel.shape[2] == mc.n_mels


def test_multires_logmels_finite(cfg):
    waveforms = torch.randn(2, cfg.audio.num_samples)
    mels = make_multires_logmels(waveforms, cfg.mel_configs, cfg.audio.sample_rate)
    for mel in mels:
        assert torch.isfinite(mel).all(), "NaN/Inf in log-mel output"


def test_multires_logmels_specaugment_applied(cfg):
    waveforms = torch.randn(4, cfg.audio.num_samples)
    make_multires_logmels(waveforms, cfg.mel_configs, cfg.audio.sample_rate, train_mode=False)
    mels_aug = make_multires_logmels(waveforms, cfg.mel_configs, cfg.audio.sample_rate, train_mode=True)
    # With augmentation some values should be zeroed
    for m_aug in mels_aug:
        assert (m_aug == 0).any()
