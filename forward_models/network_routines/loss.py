"""Loss functions and regularisers for albedo training."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _reduce(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.sum().clamp(min=1)
    return (loss * mask).sum() / valid


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _reduce(torch.abs(pred - target), mask)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _reduce((pred - target) ** 2, mask)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 0.01) -> torch.Tensor:
    diff = torch.abs(pred - target)
    quadratic = torch.clamp(diff, max=delta)
    linear = diff - quadratic
    return _reduce(0.5 * quadratic ** 2 + delta * linear, mask)


def line_core_weights(lam: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, top_fraction: float = 0.0) -> torch.Tensor:
    if top_fraction <= 0.0:
        return torch.ones_like(target)
    top_fraction = min(max(top_fraction, 0.0), 1.0)
    lam_diff = lam[:, 1:] - lam[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    valid_pairs = mask[:, 1:] & mask[:, :-1]
    grad = torch.zeros_like(target)
    safe = lam_diff.clone()
    safe = torch.where(valid_pairs, safe, torch.ones_like(safe))
    grad_values = torch.abs(target_diff / safe.clamp(min=1e-6))
    grad[:, 1:] = torch.where(valid_pairs, grad_values, torch.zeros_like(grad_values))
    grad[:, 0] = grad[:, 1]
    grad = grad * mask
    flat = grad[mask]
    if flat.numel() == 0:
        return torch.ones_like(target)
    q = torch.quantile(flat, 1.0 - top_fraction)
    weights = torch.ones_like(target)
    weights.masked_fill_(grad >= q, 2.0)
    return weights


def second_derivative_penalty(pred: torch.Tensor, lam: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    if diff_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    lam_slice = lam[:, :-2][diff_mask]
    lam_mid = lam[:, 1:-1][diff_mask]
    lam_far = lam[:, 2:][diff_mask]
    pred_slice = pred[:, :-2][diff_mask]
    pred_mid = pred[:, 1:-1][diff_mask]
    pred_far = pred[:, 2:][diff_mask]
    denom_left = (lam_mid - lam_slice).clamp(min=1e-6)
    denom_right = (lam_far - lam_mid).clamp(min=1e-6)
    left_grad = (pred_mid - pred_slice) / denom_left
    right_grad = (pred_far - pred_mid) / denom_right
    second = right_grad - left_grad
    return torch.mean(second ** 2)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    mae = masked_mae(pred, target, mask).item()
    rmse = torch.sqrt(masked_mse(pred, target, mask)).item()
    abs_err = torch.abs(pred - target) * mask
    total = mask.sum().clamp(min=1)
    delta1 = (abs_err <= 0.01).sum().float() / total
    delta2 = (abs_err <= 0.02).sum().float() / total
    delta5 = (abs_err <= 0.05).sum().float() / total
    return {
        "mae": mae,
        "rmse": rmse,
        "pct_lt_1": delta1.item() * 100.0,
        "pct_lt_2": delta2.item() * 100.0,
        "pct_lt_5": delta5.item() * 100.0,
    }
