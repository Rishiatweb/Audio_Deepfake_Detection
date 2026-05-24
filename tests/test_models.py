"""Tests for model forward passes, output shapes, and gradient flow."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.config import load_config
from src.models.aasist import AASIST
from src.models.components import (
    ConformerBlock,
    CrossScaleAttentionFusion,
    DomainDiscriminator,
    ScaleEncoder,
    grad_reverse,
)
from src.models.condetection import ConDetection
from src.models.factory import count_parameters, get_model
from src.models.lcnn import LCNN
from src.models.rawnet2 import RawNet2


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


@pytest.fixture
def dummy_mels(cfg):
    """Minimal mel batch: (2, 1, n_mels, T) per scale."""
    B = 2
    mels = []
    for mc in cfg.mel_configs:
        T = 251  # typical time frames for 4s at hop 256
        mels.append(torch.randn(B, 1, mc.n_mels, T))
    return mels


@pytest.fixture
def dummy_waveforms(cfg):
    """Raw waveform batch (2, num_samples)."""
    return torch.randn(2, cfg.audio.num_samples)


# ─── GradReverse ─────────────────────────────────────────────────────────────


def test_grad_reverse_forward_is_identity():
    x = torch.randn(4, 8, requires_grad=True)
    y = grad_reverse(x, lambda_=0.5)
    assert torch.allclose(x, y)


def test_grad_reverse_negates_gradient():
    x = torch.randn(4, 8, requires_grad=True)
    y = grad_reverse(x, lambda_=1.0)
    loss = y.sum()
    loss.backward()
    # gradient should be -1 (reversed from +1)
    assert x.grad is not None
    assert torch.allclose(x.grad, -torch.ones_like(x.grad))


# ─── DomainDiscriminator ─────────────────────────────────────────────────────


def test_domain_discriminator_output_shape():
    disc = DomainDiscriminator(in_dim=128, hidden=64)
    x = torch.randn(4, 128)
    out = disc(x, lambda_=0.3)
    assert out.shape == (4,)


# ─── ScaleEncoder ────────────────────────────────────────────────────────────


def test_scale_encoder_output_shape():
    enc = ScaleEncoder(n_mels=80, d_out=128)
    x = torch.randn(2, 1, 80, 251)
    out = enc(x)
    assert out.shape[0] == 2
    assert out.shape[2] == 128


# ─── ConformerBlock ──────────────────────────────────────────────────────────


def test_conformer_block_output_shape():
    block = ConformerBlock(d=128, heads=4)
    x = torch.randn(2, 50, 128)
    out = block(x)
    assert out.shape == x.shape


# ─── CrossScaleAttentionFusion ───────────────────────────────────────────────


def test_cross_scale_fusion_output_shape():
    fusion = CrossScaleAttentionFusion(d=128, heads=4)
    zs = [torch.randn(2, 128) for _ in range(3)]
    fused, attn_w = fusion(zs)
    assert fused.shape == (2, 128)
    assert attn_w.shape[0] == 2


# ─── ConDetection ────────────────────────────────────────────────────────────


def test_condetection_forward_shapes(cfg, dummy_mels):
    model = ConDetection(cfg)
    model.eval()
    with torch.no_grad():
        logits, cembs, domain_logits = model(dummy_mels)
    B = dummy_mels[0].shape[0]
    assert logits.shape == (B,), f"Expected ({B},), got {logits.shape}"
    assert len(cembs) == len(cfg.mel_configs)
    assert domain_logits is None  # no domain labels provided


def test_condetection_with_domain_labels(cfg, dummy_mels):
    model = ConDetection(cfg)
    model.eval()
    B = dummy_mels[0].shape[0]
    domain_labels = torch.zeros(B)
    with torch.no_grad():
        logits, cembs, domain_logits = model(dummy_mels, domain_labels=domain_labels)
    assert domain_logits is not None
    assert domain_logits.shape == (B,)


def test_condetection_consistency_loss(cfg, dummy_mels):
    model = ConDetection(cfg)
    B = dummy_mels[0].shape[0]
    domain_labels = torch.zeros(B)
    logits, cembs, _ = model(dummy_mels, domain_labels=domain_labels)
    loss = model.consistency_loss(cembs)
    assert loss.shape == ()
    assert loss >= 0


def test_condetection_no_nan_output(cfg, dummy_mels):
    model = ConDetection(cfg)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(dummy_mels)
    assert torch.isfinite(logits).all(), "NaN/Inf in ConDetection output"


def test_condetection_grad_flow(cfg, dummy_mels):
    model = ConDetection(cfg)
    model.train()
    logits, _, _ = model(dummy_mels)
    logits.sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN gradient in {name}"


# ─── AASIST ──────────────────────────────────────────────────────────────────


def test_aasist_forward_shape(cfg, dummy_mels):
    model = AASIST(cfg)
    model.eval()
    with torch.no_grad():
        logits, embs, domain = model(dummy_mels)
    assert logits.shape == (dummy_mels[0].shape[0],)
    assert embs == []
    assert domain is None


# ─── LCNN ────────────────────────────────────────────────────────────────────


def test_lcnn_forward_shape(cfg, dummy_mels):
    model = LCNN(cfg)
    model.eval()
    with torch.no_grad():
        logits, embs, domain = model(dummy_mels)
    assert logits.shape == (dummy_mels[0].shape[0],)


# ─── RawNet2 ─────────────────────────────────────────────────────────────────


def test_rawnet2_forward_shape(cfg):
    """RawNet2 takes raw waveform wrapped in mels_list[0] as (B,1,num_samples)."""
    model = RawNet2(cfg)
    model.eval()
    B = 2
    wav = torch.randn(B, 1, cfg.audio.num_samples)
    # mels_list[0] = waveform (B,1,num_samples); other scales unused
    dummy = [wav] + [torch.randn(B, 1, 80, 251)] * (len(cfg.mel_configs) - 1)
    with torch.no_grad():
        logits, embs, domain = model(dummy)
    assert logits.shape == (B,)


# ─── Model factory ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["condetection", "aasist", "lcnn"])
def test_factory_creates_model(cfg, name):
    model = get_model(name, cfg)
    assert isinstance(model, nn.Module)


def test_factory_unknown_raises(cfg):
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("unknown_model_xyz", cfg)


def test_count_parameters(cfg):
    model = get_model("condetection", cfg)
    params = count_parameters(model)
    assert params["trainable"] > 0
    assert params["total"] >= params["trainable"]
