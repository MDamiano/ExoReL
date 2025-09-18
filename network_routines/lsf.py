# PLAN:
# - Implement utilities to build a Gaussian line-spread function kernel in index space.
# - Provide torch helpers to convolve spectra batched along wavelength with optional masking.
from __future__ import annotations

from typing import Optional, Tuple

import math
import torch


def gaussian_kernel1d(sigma: float, radius_factor: float = 3.0) -> torch.Tensor:
    """Return a normalised 1D Gaussian kernel for convolution."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = max(int(radius_factor * sigma + 0.5), 1)
    idx = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (idx / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.view(1, 1, -1)


def apply_gaussian_lsf(
    spectrum: torch.Tensor,
    sigma: Optional[float],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply Gaussian LSF via conv1d while preserving batch/length dims."""
    if sigma is None:
        return spectrum
    if spectrum.dim() != 2:
        raise ValueError("spectrum must be [batch, length]")
    kernel = gaussian_kernel1d(float(sigma)).to(spectrum.device, spectrum.dtype)
    padded = spectrum.unsqueeze(1)
    out = torch.nn.functional.conv1d(padded, kernel, padding=kernel.shape[-1] // 2)
    out = out.squeeze(1)
    if mask is not None:
        out = out * mask
    return out


__all__ = ["gaussian_kernel1d", "apply_gaussian_lsf"]
