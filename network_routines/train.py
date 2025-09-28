"""Training entry-point for the ExoReL albedo transformer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from dataclasses import asdict
from contextlib import nullcontext

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split

from .data import ExoReLAlbedoDataset, pad_collate
from .loss import compute_metrics, line_core_weights, masked_mae, masked_huber, masked_mse, second_derivative_penalty
from .model import AlbedoTransformer, ModelConfig
from .utils import (
    TrainingConfig,
    cosine_schedule_with_warmup,
    ensure_output_tree,
    load_config,
    save_json,
    seed_everything,
    select_device,
    setup_logger,
)


def build_dataloaders(dataset: ExoReLAlbedoDataset, cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    val_fraction = min(max(cfg.val_fraction, 0.05), 0.5)
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset too small for the chosen validation split")
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=pad_collate,
    )
    return train_loader, val_loader


def create_model(dataset: ExoReLAlbedoDataset, device: torch.device) -> AlbedoTransformer:
    param_dim = len(dataset.param_columns)
    model_cfg = ModelConfig()
    model = AlbedoTransformer(param_dim=param_dim, config=model_cfg)
    return model.to(device)


def elementwise_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str, delta: float) -> torch.Tensor:
    if loss_type == "mae":
        return torch.abs(pred - target)
    if loss_type == "mse":
        return (pred - target) ** 2
    diff = torch.abs(pred - target)
    quadratic = torch.clamp(diff, max=delta)
    linear = diff - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def reduce_loss(loss_values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weighted_mask = mask.float() * weights
    denom = weighted_mask.sum().clamp(min=1.0)
    return (loss_values * weighted_mask).sum() / denom


def train_epoch(
    model: AlbedoTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cfg: TrainingConfig,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    batches = 0
    mixed_precision = device.type in ("cuda", "mps")
    for batch in loader:
        params = batch["p"].to(device)
        lam = batch["lam"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.autocast(device_type=device.type) if mixed_precision else nullcontext()
        with autocast_context:
            pred = model(params, lam, mask=mask)
            weights = torch.ones_like(target)
            if cfg.line_core_fraction > 0.0:
                weights = line_core_weights(lam, target, mask, cfg.line_core_fraction)
            loss_values = elementwise_loss(pred, target, cfg.loss, cfg.huber_delta)
            loss = reduce_loss(loss_values, mask, weights)
            if cfg.smoothness_weight > 0.0:
                smooth_penalty = second_derivative_penalty(pred, lam, mask)
                loss = loss + cfg.smoothness_weight * smooth_penalty
        scaler.scale(loss).backward()
        if cfg.grad_clip is not None and cfg.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            batch_mae = masked_mae(pred, target, mask).item()
        total_loss += loss.item()
        total_mae += batch_mae
        batches += 1
    return total_loss / max(1, batches), total_mae / max(1, batches)


def validate_epoch(
    model: AlbedoTransformer,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_mae = 0.0
    batches = 0
    aggregate_metrics: Dict[str, float] = {"mae": 0.0, "rmse": 0.0, "pct_lt_1": 0.0, "pct_lt_2": 0.0, "pct_lt_5": 0.0}
    with torch.no_grad():
        for batch in loader:
            params = batch["p"].to(device)
            lam = batch["lam"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            pred = model(params, lam, mask=mask)
            batch_metrics = compute_metrics(pred, target, mask)
            for key in aggregate_metrics:
                aggregate_metrics[key] += batch_metrics[key]
            total_mae += batch_metrics["mae"]
            batches += 1
    if batches > 0:
        for key in aggregate_metrics:
            aggregate_metrics[key] /= batches
        total_mae /= batches
    return total_mae, aggregate_metrics


def plot_training(history: List[Dict[str, float]], output_path: Path) -> None:
    epochs = list(range(1, len(history) + 1))
    train_mae = [row["train_mae"] for row in history]
    val_mae = [row["val_mae"] for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_mae, label="train")
    plt.plot(epochs, val_mae, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_checkpoint(
    model: AlbedoTransformer,
    optimizer: torch.optim.Optimizer,
    history: List[Dict[str, float]],
    cfg: Dict[str, any],
    path: Path,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "config": cfg,
    }
    torch.save(payload, path)


def build_checkpoint_path(checkpoint_dir: Path, out_name: Optional[str]) -> Path:
    """Resolve checkpoint filename, ensuring .pt suffix and removing directories."""

    name = (out_name or "best").strip()
    if not name:
        name = "best"
    name = Path(name).name  # prevent nested paths
    if not name.endswith(".pt"):
        name = f"{name}.pt"
    return checkpoint_dir / name


def _torch_load_checkpoint(path: Path, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resume_training_state(
    checkpoint_path: Path,
    model: AlbedoTransformer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger,
) -> Tuple[List[Dict[str, float]], float, int, int]:
    """Restore model/optimizer state from checkpoint if available."""

    if not checkpoint_path.exists():
        logger.warning("Resume requested but checkpoint '%s' not found; starting fresh.", checkpoint_path.name)
        return [], float("inf"), -1, 0

    payload = _torch_load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    optimizer_state = payload.get("optimizer")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    history_entries = [dict(row) for row in payload.get("history", []) if isinstance(row, dict)]

    best_val = float("inf")
    best_epoch = -1
    for idx, row in enumerate(history_entries):
        val = row.get("val_mae", float("inf"))
        if val < best_val:
            best_val = val
            best_epoch = idx

    start_epoch = len(history_entries)

    logger.info(
        "Resuming training from checkpoint '%s' at epoch %d.",
        checkpoint_path.name,
        start_epoch,
    )
    if best_epoch >= 0 and best_val < float("inf"):
        logger.info("Loaded best validation MAE %.5f from epoch %d.", best_val, best_epoch + 1)

    return history_entries, best_val, best_epoch, start_epoch


ConfigSource = Union[Path, str, os.PathLike, Mapping[str, Any]]


def _load_config(cfg_source: ConfigSource) -> Dict[str, Any]:
    if isinstance(cfg_source, Mapping):
        cfg = {**cfg_source}
        if "dataset_dir" not in cfg:
            raise KeyError("Configuration dictionary must include 'dataset_dir'.")
        if "output_directory" not in cfg:
            raise KeyError("Configuration dictionary must include 'output_directory'.")
        cfg.setdefault("network_training", {})
        if not isinstance(cfg["network_training"], Mapping):
            raise TypeError("'network_training' must be a mapping when provided explicitly.")
        cfg["network_training"] = dict(cfg["network_training"])
        return cfg
    path = Path(cfg_source)
    return load_config(path)

def run_training(cfg_source: ConfigSource) -> Dict[str, any]:
    cfg = _load_config(cfg_source)
    training_cfg = TrainingConfig.from_dict(cfg.get("network_training"))
    output_dirs = ensure_output_tree(cfg["output_directory"])
    output_root = output_dirs["root"]

    def _normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: _normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_normalize(v) for v in value]
        if isinstance(value, (Path, os.PathLike)):
            return os.fspath(value)
        return value

    config_snapshot = {k: _normalize(v) for k, v in cfg.items() if k != "network_training"}
    config_snapshot["output_directory"] = os.fspath(output_root)
    training_cfg_snapshot = asdict(training_cfg)
    training_cfg_snapshot["output_dir"] = os.fspath(output_root)
    config_snapshot["network_training"] = _normalize(training_cfg_snapshot)

    seed_everything(training_cfg.seed)
    device = select_device(training_cfg.device)
    logger = setup_logger(output_dirs["logs"])
    dataset = ExoReLAlbedoDataset(
        dataset_dir=cfg["dataset_dir"],
        random_lambda_fraction=training_cfg.random_lambda_fraction,
    )

    stats = dataset.parameter_stats()
    inference_stats = {
        "param_columns": list(dataset.param_columns),
        "minimum": [float(x) for x in stats.minimum.cpu().tolist()],
        "scale": [float(x) for x in stats.scale.cpu().tolist()],
    }
    config_snapshot["inference_stats"] = _normalize(inference_stats)
    save_json(config_snapshot, output_root / "network_config.json")
    train_loader, val_loader = build_dataloaders(dataset, training_cfg)
    model = create_model(dataset, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay)
    scaler = GradScaler(enabled=device.type in ("cuda", "mps"))
    schedule = cosine_schedule_with_warmup(training_cfg.epochs, training_cfg.warmup_epochs, training_cfg.lr)

    checkpoint_path = build_checkpoint_path(output_dirs["checkpoints"], cfg.get("out_net_name"))
    resume_requested = bool(cfg.get("network_training", "resume_training"))

    if resume_requested and checkpoint_path.exists():
        network_name = cfg.get("out_net_name")
        if not network_name:
            network_name = checkpoint_path.stem
        resume_msg = (
            f"Found checkpoint '{checkpoint_path.name}' for network '{network_name}'. "
            "Resuming training from this checkpoint."
        )
        print(resume_msg)
        logger.info(resume_msg)

    logger.info("Starting training for %d epochs", training_cfg.epochs)
    if resume_requested:
        history, best_val, best_epoch, start_epoch = resume_training_state(
            checkpoint_path, model, optimizer, device, logger
        )
    else:
        history = []
        best_val = float("inf")
        best_epoch = -1
        start_epoch = 0
    patience_counter = 0

    if start_epoch >= training_cfg.epochs:
        logger.info("Checkpoint already includes %d epochs, nothing to train.", start_epoch)
    for epoch in range(start_epoch, training_cfg.epochs):
        for group in optimizer.param_groups:
            group["lr"] = schedule[min(epoch, len(schedule) - 1)]
        start = time.time()
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, scaler, device, training_cfg)
        val_mae, val_metrics = validate_epoch(model, val_loader, device)
        elapsed = time.time() - start
        history.append({"train_loss": train_loss, "train_mae": train_mae, "val_mae": val_mae, **val_metrics})
        logger.info(
            "Epoch %d | train_loss=%.5f train_mae=%.5f val_mae=%.5f time=%.1fs",
            epoch + 1,
            train_loss,
            train_mae,
            val_mae,
            elapsed,
        )
        if val_mae + 1e-6 < best_val:
            best_val = val_mae
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                history,
                cfg,
                checkpoint_path,
            )
        else:
            patience_counter += 1
        if patience_counter >= training_cfg.patience:
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    plot_training(history, output_dirs["plots"] / "learning_curve.png")
    logger.info("Best validation MAE %.5f at epoch %d", best_val, best_epoch + 1)
    return {"history": history, "best_val": best_val, "best_epoch": best_epoch}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ExoReL albedo transformer")
    parser.add_argument("--par", type=str, required=True, help="Path to params.json configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(Path(args.par))


if __name__ == "__main__":
    main()
