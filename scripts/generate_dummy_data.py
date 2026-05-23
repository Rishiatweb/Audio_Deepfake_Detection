"""Generate synthetic FoR + ITW audio for pipeline testing without real data.

Real class: band-limited noise shaped to speech spectrum (formant peaks).
Fake class: same + periodic vocoder buzz (pitch pulse train) → learnable artifact.
"""
from __future__ import annotations

import argparse
import struct
import wave
from pathlib import Path

import numpy as np

SR = 16000
DURATION = 4  # seconds
N = SR * DURATION  # 64000 samples


# ─── Signal generators ────────────────────────────────────────────────────────

def _formant_filter(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shape white noise to rough speech spectrum via simple IIR resonators."""
    # Three formants in speech range: F1 ~700Hz, F2 ~1300Hz, F3 ~2500Hz
    f1 = rng.uniform(500, 900)
    f2 = rng.uniform(1100, 1600)
    f3 = rng.uniform(2000, 3000)
    out = sig.copy().astype(np.float64)
    bw = 80.0  # bandwidth Hz
    for f0 in [f1, f2, f3]:
        r = np.exp(-np.pi * bw / SR)
        theta = 2 * np.pi * f0 / SR
        a1 = -2 * r * np.cos(theta)
        a2 = r ** 2
        # Two-pole IIR resonator
        y = np.zeros_like(out)
        for i in range(2, len(out)):
            y[i] = out[i] - a1 * y[i - 1] - a2 * y[i - 2]
        out = out + 0.4 * y
    return out


def _natural_envelope(n: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth random amplitude envelope (speech-like on/off)."""
    n_segs = rng.integers(4, 10)
    seg_len = n // n_segs
    env = np.ones(n)
    for i in range(n_segs):
        if rng.random() < 0.3:  # silence segment
            s = i * seg_len
            e = min(s + seg_len, n)
            env[s:e] = rng.uniform(0.02, 0.12)
    # Smooth with running average
    kernel = np.ones(SR // 20) / (SR // 20)
    env = np.convolve(env, kernel, mode="same")
    return np.clip(env, 0.01, 1.0)


def make_real(rng: np.random.Generator) -> np.ndarray:
    """Natural speech-like signal: filtered noise with envelope."""
    noise = rng.standard_normal(N).astype(np.float32)
    shaped = _formant_filter(noise, rng).astype(np.float32)
    env = _natural_envelope(N, rng).astype(np.float32)
    sig = shaped * env
    mx = np.max(np.abs(sig))
    return (sig / mx * 0.9).astype(np.float32) if mx > 0 else sig


def make_fake(rng: np.random.Generator) -> np.ndarray:
    """Vocoder-style: real signal + pitch pulse train artifact."""
    base = make_real(rng)
    # Add periodic buzz: pulse train at F0 (TTS fundamental frequency artifact)
    f0 = rng.uniform(100, 200)  # pitch Hz
    t = np.arange(N) / SR
    # Soft pulse train via sum of harmonics
    buzz = np.zeros(N, dtype=np.float32)
    for k in range(1, 8):
        buzz += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t).astype(np.float32)
    buzz_env = _natural_envelope(N, rng).astype(np.float32)
    buzz = buzz * buzz_env * rng.uniform(0.08, 0.18)
    sig = base + buzz.astype(np.float32)
    mx = np.max(np.abs(sig))
    return (sig / mx * 0.9).astype(np.float32) if mx > 0 else sig


# ─── WAV writer ───────────────────────────────────────────────────────────────

def write_wav(path: Path, sig: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(sig * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))


# ─── Dataset builders ─────────────────────────────────────────────────────────

def make_for_split(root: Path, split: str, n_real: int, n_fake: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n_real):
        write_wav(root / split / "real" / f"{split}_real_{i:05d}.wav", make_real(rng))
    for i in range(n_fake):
        write_wav(root / split / "fake" / f"{split}_fake_{i:05d}.wav", make_fake(rng))
    print(f"  {split}: {n_real} real, {n_fake} fake")


def make_itw(root: Path, n_real: int, n_fake: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n_real):
        write_wav(root / "real" / f"itw_real_{i:05d}.wav", make_real(rng))
    for i in range(n_fake):
        write_wav(root / "fake" / f"itw_fake_{i:05d}.wav", make_fake(rng))
    print(f"  ITW: {n_real} real, {n_fake} fake")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic FoR + ITW dummy data")
    p.add_argument("--out", default="data", help="Output root directory")
    p.add_argument("--scale", type=str, default="small",
                   choices=["tiny", "small", "medium"],
                   help="tiny=500 train, small=2000, medium=5000")
    return p.parse_args()


SCALES = {
    #         train  val  test  itw
    "tiny":   (500,  150,  150,  200),
    "small":  (2000, 500,  500,  500),
    "medium": (5000, 1000, 1000, 1000),
}


def main() -> None:
    args = parse_args()
    n_train, n_val, n_test, n_itw = SCALES[args.scale]
    out = Path(args.out)

    print(f"Generating synthetic data (scale={args.scale}) -> {out.resolve()}")
    print(f"Each WAV: {SR}Hz, {DURATION}s, {N} samples\n")

    # FoR dataset — four subsets (for-original is the main one)
    subsets = {
        "for-original": "for-original",
        "for-norm":     "for-norm",
        "for-2sec":     "for-2seconds",
        "for-rerec":    "for-rerecorded",
    }

    for key, inner in subsets.items():
        subset_root = out / "for-dataset" / key / inner
        # Scale down auxiliary subsets to reduce generation time
        scale = 1.0 if key == "for-original" else 0.25
        print(f"Subset: {key}")
        make_for_split(subset_root, "training",   int(n_train * scale), int(n_train * scale), seed=1)
        make_for_split(subset_root, "validation", int(n_val * scale),   int(n_val * scale),   seed=2)
        make_for_split(subset_root, "testing",    int(n_test * scale),  int(n_test * scale),  seed=3)

    # ITW dataset
    print("\nITW:")
    make_itw(out / "in-the-wild", n_itw // 2, n_itw // 2, seed=99)

    total = sum(f.stat().st_size for f in (out).rglob("*.wav"))
    print(f"\nDone. Total size: {total / 1e6:.1f} MB")
    print("\nSet in configs/default.yaml:")
    print(f"  for_base: {(out / 'for-dataset').resolve()}")
    print(f"  itw_root: {(out / 'in-the-wild').resolve()}")


if __name__ == "__main__":
    main()
