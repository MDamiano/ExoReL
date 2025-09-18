# PLAN:
# - Define dataclasses encapsulating optimiser/training settings and seed control.
# - Build training/validation loops leveraging datasets, model, and loss helpers.
# - Provide CLI entry that saves checkpoints, metrics, and a quick validation plot.
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader

from .data import ExoReLAlbedoDataset
from .loss import masked_mse, smoothness_penalty
from .model import AlbedoTransformer, AlbedoTransformerConfig


@dataclass
class TrainingConfig:
    dataset_dir: str
    output_dir: str
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-2
    smoothness_weight: float = 0.0
    patience: int = 8
    seed: int = 42
    random_lambda_fraction: float = 1.0
    device: str = "auto"
    val_fraction: float = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device(device)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    diff = (pred - target).abs()
    diff = diff[mask]
    mae = diff.mean().item()
    rmse = math.sqrt(((pred - target)[mask] ** 2).mean().item())
    within_1 = (diff < 0.01).float().mean().item()
    within_2 = (diff < 0.02).float().mean().item()
    within_5 = (diff < 0.05).float().mean().item()
    return {
        "mae": mae,
        "rmse": rmse,
        "pct_within_1": within_1,
        "pct_within_2": within_2,
        "pct_within_5": within_5,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    smoothness_weight: float,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        params = batch["p"].to(device)
        lam = batch["lam"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        throughput = batch["throughput"].to(device)
        lsf_sigma = batch["lsf_sigma"]
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=device.type == "cuda"):
            pred = model(lam, params, throughput=throughput, lsf_sigma=lsf_sigma)
            loss = masked_mse(pred, target, mask)
            if smoothness_weight > 0:
                loss = loss + smoothness_weight * smoothness_penalty(pred, mask)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * lam.shape[0]
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float], Dict[str, torch.Tensor]]:
    model.eval()
    losses = []
    metrics_accum = []
    sample_batch: Dict[str, torch.Tensor] | None = None
    for batch in loader:
        params = batch["p"].to(device)
        lam = batch["lam"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        throughput = batch["throughput"].to(device)
        lsf_sigma = batch["lsf_sigma"]
        pred = model(lam, params, throughput=throughput, lsf_sigma=lsf_sigma)
        losses.append(masked_mse(pred, target, mask).item())
        metrics_accum.append(compute_metrics(pred, target, mask))
        if sample_batch is None:
            sample_batch = {
                "lam": lam.cpu(),
                "target": target.cpu(),
                "pred": pred.cpu(),
                "mask": mask.cpu(),
            }
    avg_metrics = {k: float(np.mean([m[k] for m in metrics_accum])) for k in metrics_accum[0]}
    return float(np.mean(losses)), avg_metrics, sample_batch or {}


def save_checkpoint(
    model: nn.Module,
    model_config: AlbedoTransformerConfig,
    config: TrainingConfig,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "train_config": asdict(config),
        "model_config": asdict(model_config),
        "metrics": metrics,
        "epoch": epoch,
    }
    torch.save(payload, path)


def plot_validation(sample: Dict[str, torch.Tensor], output_path: Path) -> None:
    if not sample:
        return
    lam = sample["lam"][0].numpy()
    target = sample["target"][0].numpy()
    pred = sample["pred"][0].numpy()
    mask = sample["mask"][0].numpy().astype(bool)
    plt.figure(figsize=(8, 4))
    plt.plot(lam[mask], target[mask], label="Target")
    plt.plot(lam[mask], pred[mask], label="Prediction")
    plt.xlabel("Wavelength")
    plt.ylabel("Albedo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train(config: TrainingConfig) -> Dict[str, float]:
    set_seed(config.seed)
    device = select_device(config.device)
    train_dataset = ExoReLAlbedoDataset(
        dataset_dir=config.dataset_dir,
        split="train",
        random_lambda_fraction=config.random_lambda_fraction,
        seed=config.seed,
        val_fraction=config.val_fraction,
    )
    val_dataset = ExoReLAlbedoDataset(
        dataset_dir=config.dataset_dir,
        split="val",
        random_lambda_fraction=None,
        seed=config.seed,
        val_fraction=config.val_fraction,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=ExoReLAlbedoDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=ExoReLAlbedoDataset.collate_fn,
    )
    model_config = AlbedoTransformerConfig(param_dim=train_dataset[0]["p"].shape[0])
    model = AlbedoTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = amp.GradScaler(enabled=device.type == "cuda")
    best_val = float("inf")
    patience_left = config.patience
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "metrics.json"
    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, config.smoothness_weight)
        val_loss, metrics, sample = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            patience_left = config.patience
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(model, model_config, config, epoch, metrics, ckpt_path)
            plot_validation(sample, output_dir / f"val_plot_epoch_{epoch}.png")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    history_path.write_text(json.dumps(history, indent=2))
    return history[-1] if history else {}


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train ExoReL Albedo Transformer")
    parser.add_argument("dataset_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-lambda-fraction", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    return TrainingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        smoothness_weight=args.smoothness_weight,
        patience=args.patience,
        seed=args.seed,
        random_lambda_fraction=args.random_lambda_fraction,
        device=args.device,
        val_fraction=args.val_fraction,
    )


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
