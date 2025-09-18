# PLAN:
# - Provide simple regression losses plus smoothness regulariser tailored for spectra.
# - Implement mean-squared error with support for padding masks.
# - Expose smoothness penalty based on second-order finite differences.
from __future__ import annotations

from typing import Optional

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    loss = torch.mean((pred - target) ** 2)
    return loss


def smoothness_penalty(spectrum: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if spectrum.shape[-1] < 3:
        return torch.tensor(0.0, device=spectrum.device, dtype=spectrum.dtype)
    second_diff = spectrum[..., :-2] - 2 * spectrum[..., 1:-1] + spectrum[..., 2:]
    if mask is not None:
        mask_mid = mask[..., 1:-1] & mask[..., :-2] & mask[..., 2:]
        second_diff = second_diff[mask_mid]
    return torch.mean(second_diff**2)


__all__ = ["masked_mse", "smoothness_penalty"]
