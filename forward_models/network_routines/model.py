"""Neural network architecture for predicting ExoReL albedo spectra."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import nn

from .fourier import FourierFeatures, FourierConfig


@dataclass
class ModelConfig:
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    param_tokens: int = 16
    self_attention: bool = True
    output_epsilon: float = 1e-4
    fourier: FourierConfig = field(default_factory=FourierConfig)


class ParamEncoder(nn.Module):
    """Project parameter vectors into a learned memory of tokens."""

    def __init__(self, in_dim: int, config: ModelConfig):
        super().__init__()
        self.tokens = config.param_tokens
        hidden = max(config.d_model, in_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.tokens * config.d_model),
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.d_model = config.d_model

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        batch = params.shape[0]
        encoded = self.encoder(params)
        encoded = encoded.view(batch, self.tokens, self.d_model)
        encoded = self.norm(encoded)
        return self.dropout(encoded)


class LambdaEmbed(nn.Module):
    """Embed Fourier features into the transformer hidden space."""

    def __init__(self, feature_dim: int, config: ModelConfig):
        super().__init__()
        hidden = config.d_model * 2
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model),
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedded = self.net(features)
        return self.dropout(self.norm(embedded))


class CrossAttentionBlock(nn.Module):
    """Pre-norm block with optional self-attention followed by cross-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model = config.d_model
        self.self_attention = config.self_attention
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.ln_cross_q = nn.LayerNorm(d_model)
        self.ln_cross_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        lam_tokens: torch.Tensor,
        param_tokens: torch.Tensor,
        lam_padding_mask: Optional[torch.Tensor],
        param_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.self_attention:
            residual = lam_tokens
            tokens = self.ln_self(lam_tokens)
            attn_out, _ = self.self_attn(tokens, tokens, tokens, key_padding_mask=lam_padding_mask)
            lam_tokens = residual + attn_out
        residual = lam_tokens
        q = self.ln_cross_q(lam_tokens)
        kv = self.ln_cross_kv(param_tokens)
        attn_out, _ = self.cross_attn(
            q,
            kv,
            kv,
            key_padding_mask=param_padding_mask,
            attn_mask=None,
        )
        lam_tokens = residual + attn_out
        lam_tokens = lam_tokens + self.ff(lam_tokens)
        return lam_tokens, param_tokens


class AlbedoTransformer(nn.Module):
    """Perceiver-style model mapping parameters and wavelengths to albedo spectra."""

    def __init__(
        self,
        param_dim: int,
        config: Optional[ModelConfig] = None,
        fourier: Optional[FourierFeatures] = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.fourier = fourier or FourierFeatures(self.config.fourier)
        self.param_encoder = ParamEncoder(param_dim, self.config)
        feature_dim = 1 + 2 * self.config.fourier.n_frequencies
        self.lambda_embed = LambdaEmbed(feature_dim, self.config)
        self.layers = nn.ModuleList([CrossAttentionBlock(self.config) for _ in range(self.config.num_layers)])
        hidden = max(32, self.config.d_model // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.output_epsilon = self.config.output_epsilon

    def forward(
        self,
        params: torch.Tensor,
        lam: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        param_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.fourier.is_fitted:
            with torch.no_grad():
                lam_flat = lam[mask] if mask is not None else lam.reshape(-1)
                self.fourier.fit(lam_flat.detach().cpu())
        features = self.fourier.transform(lam, device=lam.device, dtype=lam.dtype)
        lam_tokens = self.lambda_embed(features)
        param_tokens = self.param_encoder(params)
        lam_padding_mask = (~mask) if mask is not None else None
        if param_mask is None:
            param_mask = torch.zeros(
                (params.shape[0], param_tokens.shape[1]), dtype=torch.bool, device=params.device
            )
        for layer in self.layers:
            lam_tokens, param_tokens = layer(
                lam_tokens,
                param_tokens,
                lam_padding_mask=lam_padding_mask,
                param_padding_mask=param_mask,
            )
        logits = self.head(lam_tokens).squeeze(-1)
        albedo = self.sigmoid(logits)
        if self.output_epsilon > 0.0:
            albedo = torch.clamp(albedo, self.output_epsilon, 1.0 - self.output_epsilon)
        self.fourier.step()
        return albedo
