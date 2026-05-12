"""Generic training loop reused by all PyTorch models.

Keeping training logic in one place ensures the custom CNN, ResNet50, and
ablation runs all use identical optimisation, scheduling, and early-stopping
behaviour. The only thing that varies is the model and the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.metrics import ClassificationMetrics, compute_metrics


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # "adam" | "sgd"
    scheduler: str = "plateau"  # "plateau" | "cosine" | "none"
    patience: int = 3  # early stopping patience (epochs without val-loss improvement)
    grad_clip: float | None = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _build_optimizer(params, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")


def _build_scheduler(opt: torch.optim.Optimizer, cfg: TrainConfig, total_epochs: int):
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=2
        )
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)
    return None


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    grad_clip: float | None,
    desc: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one epoch. If ``optimizer`` is None, runs in eval mode."""
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            losses.append(loss.item())
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.detach().cpu().numpy())

    return (
        float(np.mean(losses)),
        np.concatenate(all_probs),
        np.concatenate(all_labels),
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    checkpoint_path: str | Path,
    on_epoch_end=None,  # callback(epoch, metrics_dict) — used for wandb/mlflow
) -> dict:
    """Train a model with early stopping and return the best-epoch summary.

    The model state corresponding to the lowest validation loss is saved to
    ``checkpoint_path``. The returned dict contains per-epoch history plus
    best-epoch metrics.
    """
    device = cfg.device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = _build_optimizer(trainable, cfg)
    scheduler = _build_scheduler(optimizer, cfg, cfg.epochs)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_metrics: ClassificationMetrics | None = None
    epochs_without_improvement = 0

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        train_loss, _, _ = _epoch_pass(
            model, train_loader, criterion, optimizer, device, cfg.grad_clip,
            desc=f"epoch {epoch}/{cfg.epochs} [train]",
        )
        val_loss, val_probs, val_labels = _epoch_pass(
            model, val_loader, criterion, None, device, None,
            desc=f"epoch {epoch}/{cfg.epochs} [val]",
        )
        val_metrics = compute_metrics(val_labels, val_probs)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.to_dict().items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_summary)
        print(
            f"[epoch {epoch:2d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_metrics.accuracy:.4f}  val_auc={val_metrics.roc_auc:.4f}"
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_summary)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics.to_dict(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"[early stop] no val-loss improvement for {cfg.patience} epochs.")
                break

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics.to_dict() if best_metrics else None,
        "checkpoint": str(checkpoint_path),
    }
