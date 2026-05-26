"""ConDetection-DANN: main model class."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.models.components import (
    ConformerBlock,
    CrossScaleAttentionFusion,
    DomainDiscriminator,
    ScaleEncoder,
    mixstyle_1d,
    sinusoidal_timestep_embedding,
)


class ConDetection(nn.Module):
    """ConDetection-DANN: Hierarchical Multi-Scale Conformer with DANN.

    Architecture:
        ScaleEncoder × K (one per mel scale)
        → optional MixStyle domain aug
        → shared Conformer blocks (n_layers)
        → cross-scale attention fusion
        → classification head (binary logit: real/fake)
        → DomainDiscriminator via GRL (source FoR vs target ITW)

    The DANN objective forces the encoder to be domain-confusing so the
    classifier relies only on forgery artifacts, not domain cues.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        mc = cfg.model
        dc = cfg.dann
        mel_cfgs = cfg.mel_configs
        K = len(mel_cfgs)

        self.use_diffusion = cfg.diffusion.enabled
        self.use_dann = dc.enabled
        self._dann_lambda: float = 0.0  # annealed externally by trainer

        # ─── Per-scale CNN encoders ───
        self.scale_encoders = nn.ModuleList([ScaleEncoder(m.n_mels, mc.d_model) for m in mel_cfgs])

        # ─── Shared Conformer blocks ───
        self.conformer = nn.ModuleList(
            [ConformerBlock(mc.d_model, heads=mc.n_heads, dropout=mc.dropout) for _ in range(mc.n_layers)]
        )
        self.pool_positions: set[int] = {1, 3}

        # ─── Cross-scale attention fusion ───
        self.cross_scale = CrossScaleAttentionFusion(mc.d_model)
        self.consist_proj = nn.ModuleList([nn.Linear(mc.d_model, mc.d_model) for _ in range(K)])

        # ─── Classification head ───
        self.classifier = nn.Sequential(
            nn.LayerNorm(mc.d_model),
            nn.Linear(mc.d_model, 64),
            nn.GELU(),
            nn.Dropout(mc.dropout),
            nn.Linear(64, 1),
        )

        # ─── DANN domain discriminator ───
        if self.use_dann:
            self.domain_disc = DomainDiscriminator(in_dim=mc.d_model, hidden=mc.dann_hidden)

        # ─── Optional diffusion branch ───
        if self.use_diffusion:
            diff_cfg = cfg.diffusion
            betas = torch.linspace(diff_cfg.beta_min, diff_cfg.beta_max, diff_cfg.steps)
            alpha_bars = torch.cumprod(1.0 - betas, dim=0)
            self.register_buffer("diff_alpha_bars", alpha_bars)
            self.time_mlp = nn.Sequential(
                nn.Linear(mc.d_model, mc.d_model), nn.SiLU(), nn.Linear(mc.d_model, mc.d_model)
            )
            self.diff_refine = nn.Sequential(
                nn.LayerNorm(mc.d_model),
                nn.Linear(mc.d_model, mc.d_model * 2),
                nn.SiLU(),
                nn.Linear(mc.d_model * 2, mc.d_model),
            )
            self.diff_gate = nn.Parameter(torch.tensor(0.15))

        self._last_diff_loss: torch.Tensor = torch.tensor(0.0)
        self._last_attn_weights: torch.Tensor | None = None

    def _pool_time(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)

    def _diffuse_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        diff_steps = self.cfg.diffusion.steps
        t_idx = torch.randint(0, diff_steps, (B,), device=x.device)
        a_bar = self.diff_alpha_bars[t_idx].view(B, 1, 1).clamp(1e-5, 0.9999)  # type: ignore[index]
        eps = torch.randn_like(x)
        x_noisy = a_bar.sqrt() * x + (1.0 - a_bar).sqrt() * eps
        t_norm = t_idx.float() / float(max(diff_steps - 1, 1))
        t_emb = self.time_mlp(sinusoidal_timestep_embedding(t_norm, x.size(-1))).unsqueeze(1)
        return x_noisy, t_emb

    def forward(
        self,
        mels_list: list[torch.Tensor],
        domain_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        """Forward pass.

        Args:
            mels_list: list of K tensors (B, 1, n_mels, T)
            domain_labels: (B,) float32 — 0=source(FoR), 1=target(ITW)

        Returns:
            logits: (B,)
            consist_embs: list of K normalised embeddings (B, D)
            domain_logits: (B,) or None
        """
        embeddings: list[torch.Tensor] = []
        diff_losses: list[torch.Tensor] = []

        for mel, encoder in zip(mels_list, self.scale_encoders):
            x_clean: torch.Tensor = encoder(mel)

            if self.training:
                x_clean = mixstyle_1d(x_clean, self.cfg.domain_gen.mixstyle_p, self.cfg.domain_gen.mixstyle_alpha)

            if self.training and self.use_diffusion:
                x, t_emb = self._diffuse_latent(x_clean)
                x = x + torch.tanh(self.diff_gate) * t_emb
                x_ref: torch.Tensor | None = x_clean
            else:
                x = x_clean
                x_ref = None

            for i, block in enumerate(self.conformer):
                x = block(x)
                if i in self.pool_positions:
                    x = self._pool_time(x)
                    if x_ref is not None:
                        x_ref = self._pool_time(x_ref)

            if x_ref is not None:
                denoised = self.diff_refine(x)
                diff_losses.append(F.smooth_l1_loss(denoised, x_ref.detach()))

            embeddings.append(x.mean(dim=1))

        fused, attn_weights = self.cross_scale(embeddings)
        self._last_attn_weights = attn_weights

        self._last_diff_loss = torch.stack(diff_losses).mean() if diff_losses else fused.new_zeros(())

        logits: torch.Tensor = self.classifier(fused).squeeze(-1)
        consist_embs = [F.normalize(self.consist_proj[k](e), dim=-1) for k, e in enumerate(embeddings)]

        domain_logits: torch.Tensor | None = None
        if self.use_dann and domain_labels is not None:
            domain_logits = self.domain_disc(fused, lambda_=self._dann_lambda)

        return logits, consist_embs, domain_logits

    def consistency_loss(self, consist_embs: list[torch.Tensor]) -> torch.Tensor:
        """Mean pairwise L2 distance between normalised scale embeddings."""
        K = len(consist_embs)
        loss = consist_embs[0].new_zeros(())
        pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                loss = loss + ((consist_embs[i] - consist_embs[j]) ** 2).sum(dim=-1).mean()
                pairs += 1
        return loss / max(pairs, 1)


# Alias used in cross-validation cells
HMConformerXAI = ConDetection
