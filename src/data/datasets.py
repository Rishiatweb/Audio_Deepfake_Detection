"""Dataset classes and data loading utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import Config
from src.data.augment import apply_waveform_augmentations


def load_audio(path: str, sample_rate: int = 16000, num_samples: int = 64000, random_crop: bool = False) -> np.ndarray:
    """Load mono audio and return fixed-length waveform as float32.

    Uses soundfile (fast) with torchaudio resampling fallback.
    Crops (center or random) if too long, zero-pads if too short.
    Peak-normalises to [-1, 1]. Returns zeros on error.
    """
    try:
        import soundfile as sf

        y, sr = sf.read(path, dtype="float32", always_2d=False)
        # Mix down to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)
        # Resample if needed (FoR/ITW are already 16kHz — skip in practice)
        if sr != sample_rate:
            import torchaudio.functional as TAF  # noqa: PLC0415
            y = TAF.resample(torch.from_numpy(y), sr, sample_rate).numpy()
        if len(y) > num_samples:
            start = random.randint(0, len(y) - num_samples) if random_crop else max((len(y) - num_samples) // 2, 0)
            y = y[start : start + num_samples]
        elif len(y) < num_samples:
            y = np.pad(y, (0, num_samples - len(y)))
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        mx = np.max(np.abs(y))
        if np.isfinite(mx) and mx > 0:
            y = y / mx
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception:
        return np.zeros(num_samples, dtype=np.float32)


def collect_audio_files(folder: str | Path, label: int, source: str, subset: str = "") -> list[dict]:
    """Recursively find all .wav files under folder."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return [{"filepath": str(fp), "label": label, "source": source, "subset": subset} for fp in folder.rglob("*.wav")]


def build_for_dataframes(for_base: str) -> dict[str, list[dict]]:
    """Collect FoR dataset splits from all four subsets."""
    subsets = {
        "for-original": "for-original",
        "for-norm": "for-norm",
        "for-2sec": "for-2seconds",
        "for-rerec": "for-rerecorded",
    }
    dfs: dict[str, list[dict]] = {}
    for key, inner in subsets.items():
        root = Path(for_base) / key / inner
        if not root.exists():
            continue
        for split in ["training", "validation", "testing"]:
            for lbl, name in [(0, "real"), (1, "fake")]:
                rows = collect_audio_files(root / split / name, lbl, key, split)
                dfs.setdefault(key, []).extend(rows)
    return dfs


def build_itw_dataframe(itw_root: str) -> pd.DataFrame:
    """Collect In-the-Wild dataset (real/fake subdirs)."""
    rows: list[dict] = []
    rows.extend(collect_audio_files(Path(itw_root) / "real", 0, "ITW", "test"))
    rows.extend(collect_audio_files(Path(itw_root) / "fake", 1, "ITW", "test"))
    return pd.DataFrame(rows)


def stratified_cap(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    """Sample up to n rows from df preserving class ratios."""
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    parts = []
    for lbl in sorted(df["label"].unique().tolist()):
        part = df[df["label"] == lbl]
        take = max(1, int(round(n * len(part) / len(df))))
        parts.append(part.sample(min(take, len(part)), random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out.iloc[:n].reset_index(drop=True)


def source_balanced_cap(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    """Sample up to n rows with balanced allocation across source subsets."""
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    sources = sorted(df["source"].unique().tolist())
    if not sources:
        return stratified_cap(df, n, seed)

    per_source = max(1, n // len(sources))
    parts = []
    for src in sources:
        part = df[df["source"] == src]
        per_class = per_source // 2
        sub_parts = []
        for lbl in sorted(part["label"].unique().tolist()):
            p = part[part["label"] == lbl]
            sub_parts.append(p.sample(min(per_class, len(p)), random_state=seed))
        parts.append(pd.concat(sub_parts) if sub_parts else part.head(0))

    out = pd.concat(parts).drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    if len(out) < n:
        extra = df[~df["filepath"].isin(set(out["filepath"]))]
        if len(extra) > 0:
            out = pd.concat([out, extra.sample(min(n - len(out), len(extra)), random_state=seed)]).reset_index(
                drop=True
            )
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True).iloc[:n].reset_index(drop=True)


def build_splits(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train/val/for_test/itw DataFrames from config paths."""
    dfs = build_for_dataframes(cfg.paths.for_base)
    itw_df = build_itw_dataframe(cfg.paths.itw_root)

    train_keys = ["for-original", "for-norm", "for-rerec", "for-2sec"]
    test_keys = ["for-original", "for-norm"]

    train_rows, val_rows, test_rows = [], [], []
    for key in train_keys:
        if key not in dfs:
            continue
        df_k = pd.DataFrame(dfs[key])
        train_rows.extend(df_k[df_k["subset"] == "training"].to_dict("records"))
        val_rows.extend(df_k[df_k["subset"] == "validation"].to_dict("records"))

    for key in test_keys:
        if key not in dfs:
            continue
        df_k = pd.DataFrame(dfs[key])
        test_rows.extend(df_k[df_k["subset"] == "testing"].to_dict("records"))

    train_df = source_balanced_cap(pd.DataFrame(train_rows), cfg.training.max_train_samples, cfg.training.seed)
    val_df = stratified_cap(pd.DataFrame(val_rows), cfg.training.max_val_samples, cfg.training.seed)
    for_test_df = pd.DataFrame(test_rows).reset_index(drop=True)

    return train_df, val_df, for_test_df, itw_df


class FastAudioDataset(Dataset):
    """Loads raw waveforms; log-mel extraction done in batches on GPU."""

    def __init__(self, df: pd.DataFrame, augment: bool = False, cfg: Config | None = None) -> None:
        self.paths = df["filepath"].tolist() if "filepath" in df.columns else []
        self.labels = df["label"].astype(np.float32).to_numpy() if "label" in df.columns else np.array([], dtype=np.float32)
        self.augment = augment
        self.num_samples = cfg.audio.num_samples if cfg else 64000

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        lbl = float(self.labels[idx])
        y = load_audio(self.paths[idx], num_samples=self.num_samples, random_crop=self.augment)
        if self.augment:
            y = apply_waveform_augmentations(y, lbl, self.num_samples)
        return torch.from_numpy(y), torch.tensor(lbl, dtype=torch.float32)


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    for_test_df: pd.DataFrame,
    itw_df: pd.DataFrame,
    cfg: Config,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Build all four DataLoaders."""
    tr_cfg = cfg.training

    train_ds = FastAudioDataset(train_df, augment=True, cfg=cfg)
    val_ds = FastAudioDataset(val_df, augment=False, cfg=cfg)
    for_test_ds = FastAudioDataset(for_test_df, augment=False, cfg=cfg)
    itw_ds = FastAudioDataset(itw_df, augment=False, cfg=cfg)

    worker_kw: dict = dict(
        num_workers=tr_cfg.num_workers,
        pin_memory=(tr_cfg.num_workers > 0),
        persistent_workers=(tr_cfg.num_workers > 0),
    )

    sampler = None
    if tr_cfg.use_balanced_sampler and len(train_df) > 0:
        cls_counts = train_df["label"].value_counts().to_dict()
        w_real = 1.0 / max(cls_counts.get(0, 1), 1)
        w_fake = 1.0 / max(cls_counts.get(1, 1), 1)
        sample_w = np.where(train_df["label"].to_numpy() == 1, w_fake, w_real).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w),
            num_samples=len(sample_w),
            replacement=True,
            generator=torch.Generator().manual_seed(tr_cfg.seed),
        )

    if len(train_ds) > 0:
        train_loader = DataLoader(
            train_ds,
            batch_size=tr_cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            **worker_kw,
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=tr_cfg.batch_size)
    val_loader = DataLoader(val_ds, batch_size=tr_cfg.batch_size * 2, shuffle=False, drop_last=False, **worker_kw) if len(val_ds) > 0 else DataLoader(val_ds, batch_size=tr_cfg.batch_size * 2)
    for_test_loader = DataLoader(
        for_test_ds, batch_size=tr_cfg.batch_size * 2, shuffle=False, drop_last=False, **worker_kw
    )
    itw_loader = DataLoader(itw_ds, batch_size=tr_cfg.batch_size * 2, shuffle=False, drop_last=False, **worker_kw)

    return train_loader, val_loader, for_test_loader, itw_loader
