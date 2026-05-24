"""ConDetection-DANN building blocks."""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── MixStyle ────────────────────────────────────────────────────────────────


def mixstyle_1d(x: torch.Tensor, p: float = 0.5, alpha: float = 0.6, eps: float = 1e-6) -> torch.Tensor:
    """MixStyle over temporal features (B, T, D) for domain generalization.

    Reference: Zhou et al., 2021 — Domain Generalization with MixStyle.
    """
    if (not torch.is_grad_enabled()) or (not x.requires_grad) or x.size(0) < 2:
        return x
    if random.random() > p:
        return x

    mu = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    sig = (var + eps).sqrt()
    x_norm = (x - mu) / sig

    perm = torch.randperm(x.size(0), device=x.device)
    mu2, sig2 = mu[perm], sig[perm]

    lam = torch.distributions.Beta(alpha, alpha).sample((x.size(0), 1, 1)).to(x.device)
    mu_mix = lam * mu + (1.0 - lam) * mu2
    sig_mix = lam * sig + (1.0 - lam) * sig2
    return x_norm * sig_mix + mu_mix


# ─── DANN: Gradient Reversal Layer ───────────────────────────────────────────


class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin et al. 2016).

    Forward: identity. Backward: negate gradient scaled by lambda_.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        (lambda_,) = ctx.saved_tensors
        return -lambda_.item() * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    """Binary domain classifier: source (FoR=0) vs target (ITW=1).

    Trained adversarially via GRL to force domain-invariant encoder features.
    """

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feat: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        """feat: (B, D) — domain label: 0=source, 1=target"""
        rev = grad_reverse(feat, lambda_)
        return self.net(rev).squeeze(-1)


# ─── Conformer building blocks ───────────────────────────────────────────────


class FeedForward(nn.Module):
    def __init__(self, d: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d * expansion, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvModule(nn.Module):
    """Conformer convolution module with GLU gating."""

    def __init__(self, d: int, ks: int = 15) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, d * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d, d, ks, padding=ks // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.norm(x).transpose(1, 2)
        r = self.glu(self.pw1(r))
        r = F.silu(self.bn(self.dw(r)))
        return self.drop(self.pw2(r)).transpose(1, 2)


class ConformerBlock(nn.Module):
    """Macaron-style Conformer block: FF½ → Attn → Conv → FF½ → LN."""

    def __init__(self, d: int, heads: int = 4, ks: int = 15, dropout: float = 0.1) -> None:
        super().__init__()
        self.ff1 = FeedForward(d, dropout=dropout)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(d)
        self.conv = ConvModule(d, ks)
        self.ff2 = FeedForward(d, dropout=dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        a, _ = self.attn(self.norm_a(x), self.norm_a(x), self.norm_a(x), need_weights=False)
        x = x + a
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class ScaleEncoder(nn.Module):
    """Lightweight CNN encoder that maps one mel-spectrogram scale to sequence embeddings."""

    def __init__(self, n_mels: int, d_out: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        freq_out = math.ceil(math.ceil(n_mels / 2) / 2)
        self.proj = nn.Linear(64 * freq_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T) → (B, T, d_out)"""
        h = self.cnn(x)  # (B, 64, F', T)
        B, C, Fm, T = h.shape
        h = h.permute(0, 3, 1, 2).reshape(B, T, C * Fm)
        return self.proj(h)


class CrossScaleAttentionFusion(nn.Module):
    """Cross-scale attention: each scale attends to all other scales."""

    def __init__(self, d: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d)

    def forward(self, zs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """zs: list of K tensors each (B, D) → fused (B, D), attn_weights (B, K, K)"""
        Z = torch.stack(zs, dim=1)  # (B, K, D)
        attn_out, attn_w = self.attn(self.norm(Z), self.norm(Z), self.norm(Z))
        fused = Z + attn_out
        return self.proj(fused.mean(dim=1)), attn_w


# ─── Diffusion helpers (optional, disabled by default) ───────────────────────


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for a normalized timestep t in [0, 1]."""
    half = dim // 2
    if half <= 0:
        return t.unsqueeze(-1)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(10000.0) / max(half - 1, 1)))
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb
