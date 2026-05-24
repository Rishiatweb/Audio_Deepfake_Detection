"""Configuration dataclasses and YAML loader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class MelConfig:
    name: str
    n_fft: int
    hop_length: int
    n_mels: int


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    duration: int = 4
    num_samples: int = 64000


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.15
    dann_hidden: int = 128


@dataclass
class DannConfig:
    enabled: bool = True
    lambda_max: float = 0.3
    warmup_epochs: int = 4


@dataclass
class DiffusionConfig:
    enabled: bool = False
    steps: int = 200
    beta_min: float = 1e-4
    beta_max: float = 0.02
    lambda_diff: float = 0.0
    warmup_epochs: int = 6


@dataclass
class DomainGenConfig:
    mixstyle_p: float = 0.5
    mixstyle_alpha: float = 0.6


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    epochs: int = 20
    patience: int = 8
    num_workers: int = 0
    max_train_samples: int = 20000
    max_val_samples: int = 5000
    max_train_steps: int | None = 400
    max_val_steps: int | None = 120
    seed: int = 42
    use_balanced_sampler: bool = True
    use_focal_loss: bool = True
    focal_gamma: float = 1.5
    focal_alpha: float = 0.60
    lambda_c: float = 0.1
    label_smooth: float = 0.03
    use_tta: bool = True
    tta_shifts: list = field(default_factory=lambda: [0, 2000, -2000])


@dataclass
class PathsConfig:
    output_dir: str = "results"
    checkpoint_dir: str = "results/checkpoints"
    figures_dir: str = "results/figures"
    tables_dir: str = "results/tables"
    logs_dir: str = "results/logs"
    for_base: str = "data/for-dataset"
    itw_root: str = "data/in-the-wild"


@dataclass
class CrossValConfig:
    n_folds: int = 5
    cv_epochs: int = 10
    cv_max_train_steps: int | None = None


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    mel_configs: list[MelConfig] = field(
        default_factory=lambda: [
            MelConfig("fine", 400, 160, 64),
            MelConfig("mid", 1024, 256, 80),
            MelConfig("coarse", 2048, 512, 128),
        ]
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    dann: DannConfig = field(default_factory=DannConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    domain_gen: DomainGenConfig = field(default_factory=DomainGenConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    cross_val: CrossValConfig = field(default_factory=CrossValConfig)

    def make_dirs(self) -> None:
        for d in [
            self.paths.output_dir,
            self.paths.checkpoint_dir,
            self.paths.figures_dir,
            self.paths.tables_dir,
            self.paths.logs_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def override_paths_from_env(self) -> None:
        """Allow environment variables to override dataset paths."""
        if p := os.environ.get("FOR_BASE"):
            self.paths.for_base = p
        if p := os.environ.get("ITW_ROOT"):
            self.paths.itw_root = p


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load Config from YAML file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    if audio := raw.get("audio"):
        cfg.audio = AudioConfig(**audio)

    if mel_list := raw.get("mel_configs"):
        cfg.mel_configs = [MelConfig(**m) for m in mel_list]

    if model := raw.get("model"):
        cfg.model = ModelConfig(**model)

    if dann := raw.get("dann"):
        cfg.dann = DannConfig(**dann)

    if diff := raw.get("diffusion"):
        cfg.diffusion = DiffusionConfig(**diff)

    if dg := raw.get("domain_generalization"):
        cfg.domain_gen = DomainGenConfig(**dg)

    if tr := raw.get("training"):
        cfg.training = TrainingConfig(**tr)

    if paths := raw.get("paths"):
        cfg.paths = PathsConfig(**paths)

    if cv := raw.get("cross_validation"):
        cfg.cross_val = CrossValConfig(**cv)

    cfg.override_paths_from_env()
    return cfg
