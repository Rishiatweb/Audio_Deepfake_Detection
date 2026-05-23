"""Model factory: create any supported model by name."""

from __future__ import annotations

import torch.nn as nn

from src.config import Config

_MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, cfg: Config) -> nn.Module:
    """Instantiate a model by name.

    Supported names:
        condetection  — ConDetection-DANN (main proposed model)
        aasist        — AASIST baseline
        rawnet2       — RawNet2 baseline
        lcnn          — LCNN baseline

    Raises:
        ValueError if name is not registered.
    """
    # Lazy imports to avoid circular deps and heavy imports at module load
    if name == "condetection":
        from src.models.condetection import ConDetection

        return ConDetection(cfg)

    if name == "aasist":
        from src.models.aasist import AASIST

        return AASIST(cfg)

    if name == "rawnet2":
        from src.models.rawnet2 import RawNet2

        return RawNet2(cfg)

    if name == "lcnn":
        from src.models.lcnn import LCNN

        return LCNN(cfg)

    available = ["condetection", "aasist", "rawnet2", "lcnn"]
    raise ValueError(f"Unknown model '{name}'. Available: {available}")


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
