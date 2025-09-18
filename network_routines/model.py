# PLAN:
# - Define configuration dataclasses and lightweight building blocks for the albedo transformer.
# - Implement ParamEncoder and LambdaEmbed modules to turn physical inputs into tokens.
# - Stack Pre-LN cross-attention transformer blocks culminating in sigmoid spectrum predictions.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .fourier import FourierFeatures
from .lsf import apply_gaussian_lsf


@dataclass
class AlbedoTransformerConfig:
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    param_tokens: int = 4
    param_dim: int = 16
    dropout: float = 0.1
    use_self_attn: bool = True
    k_fourier: int = 16


class ParamEncoder(nn.Module):
    def __init__(self, config: AlbedoTransformerConfig):
        super().__init__()
        self.param_tokens = config.param_tokens
        hidden = config.d_model * 2
        self.net = nn.Sequential(
            nn.LayerNorm(config.param_dim),
            nn.Linear(config.param_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model * config.param_tokens),
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        batch = params.shape[0]
        tokens = self.net(params).view(batch, self.param_tokens, -1)
        return tokens


class LambdaEmbed(nn.Module):
    def __init__(self, config: AlbedoTransformerConfig):
        super().__init__()
        self.k_fourier = config.k_fourier
        dummy_ff = FourierFeatures(num_frequencies=self.k_fourier)
        proj_in = dummy_ff.output_dim
        self.proj = nn.Sequential(
            nn.Linear(proj_in, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, lam: torch.Tensor) -> torch.Tensor:
        if lam.dim() != 2:
            raise ValueError("lambda input must be [batch, length]")
        batch, length = lam.shape
        device = lam.device
        embeddings = []
        for b in range(batch):
            lam_b = lam[b]
            ff = FourierFeatures(num_frequencies=self.k_fourier)
            ff.fit(lam_b)
            emb = ff.transform(lam_b).to(device)
            embeddings.append(emb)
        stacked = torch.stack(embeddings, dim=0)
        return self.proj(stacked)


class CrossAttnBlock(nn.Module):
    def __init__(self, config: AlbedoTransformerConfig):
        super().__init__()
        self.use_self = config.use_self_attn
        self.ln_self = nn.LayerNorm(config.d_model)
        self.ln_cross = nn.LayerNorm(config.d_model)
        self.ln_ff = nn.LayerNorm(config.d_model)
        self.self_attn = (
            nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            if self.use_self
            else None
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, lam_tokens: torch.Tensor, param_tokens: torch.Tensor) -> torch.Tensor:
        x = lam_tokens
        if self.use_self and self.self_attn is not None:
            residual = x
            x_norm = self.ln_self(x)
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
            x = residual + self.dropout(attn_out)
        residual = x
        x_norm = self.ln_cross(x)
        attn_out, _ = self.cross_attn(x_norm, param_tokens, param_tokens, need_weights=False)
        x = residual + self.dropout(attn_out)
        residual = x
        x_norm = self.ln_ff(x)
        x = residual + self.ff(x_norm)
        return x


class AlbedoTransformer(nn.Module):
    def __init__(self, config: Optional[AlbedoTransformerConfig] = None):
        super().__init__()
        self.config = config or AlbedoTransformerConfig()
        self.param_encoder = ParamEncoder(self.config)
        self.lambda_embed = LambdaEmbed(self.config)
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, 1),
        )

    def forward(
        self,
        lam: torch.Tensor,
        params: torch.Tensor,
        throughput: Optional[torch.Tensor] = None,
        lsf_sigma: Optional[float] = None,
    ) -> torch.Tensor:
        lam_tokens = self.lambda_embed(lam)
        param_tokens = self.param_encoder(params)
        x = lam_tokens
        for block in self.blocks:
            x = block(x, param_tokens)
        logits = self.head(x).squeeze(-1)
        spectrum = torch.sigmoid(logits)
        if throughput is not None:
            spectrum = spectrum * throughput
        spectrum = apply_gaussian_lsf(spectrum, lsf_sigma)
        return spectrum.clamp(0.0, 1.0)


__all__ = [
    "AlbedoTransformer",
    "AlbedoTransformerConfig",
    "ParamEncoder",
    "LambdaEmbed",
    "CrossAttnBlock",
]
