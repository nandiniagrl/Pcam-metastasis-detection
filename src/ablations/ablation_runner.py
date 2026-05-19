"""Shared utilities for ablation experiments.

An "ablation" here means systematically varying one design choice (e.g.
augmentation on/off, learning rate, freezing depth) while keeping everything
else fixed, then comparing test-set performance. All ablations log to MLflow
under the ``pcam-ablations`` experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PCamDataset
from src.data.transforms import build_eval_transform, build_train_transform
from src.evaluation.evaluate import evaluate_and_save
from src.models.resnet50 import ResNet50Classifier
from src.training.trainer import TrainConfig, train
from src.utils.logger import ExperimentLogger
from src.utils.seed import seed_worker, set_seed


@dataclass
class AblationConfig:
    """Single configuration in an ablation grid."""

    name: str
    augment: bool = True
    lr: float = 1e-4
    freeze: str = "last_block"
    optimizer: str = "adam"
    epochs: int = 5
    batch_size: int = 128
    weight_decay: float = 1e-4


def run_single(
    cfg: AblationConfig,
    data_root: str,
    seed: int,
    output_dir: Path,
    subset_fraction: float = 1.0,
    num_workers: int = 4,
    use_wandb: bool = False,
) -> dict:
    """Train ResNet50 with the given AblationConfig and return test-set metrics."""
    set_seed(seed)

    train_ds = PCamDataset(data_root, "train", build_train_transform(cfg.augment),
                           subset_fraction=subset_fraction)
    val_ds = PCamDataset(data_root, "valid", build_eval_transform())
    test_ds = PCamDataset(data_root, "test", build_eval_transform())

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    model = ResNet50Classifier(freeze=cfg.freeze, pretrained=True)
    train_cfg = TrainConfig(
        epochs=cfg.epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        optimizer=cfg.optimizer, scheduler="plateau", patience=2,
    )
    ckpt = output_dir / f"{cfg.name}_seed{seed}.pt"

    run_name = f"{cfg.name}_seed{seed}"
    log_cfg = {**cfg.__dict__, "seed": seed}
    with ExperimentLogger(run_name, log_cfg, use_wandb=use_wandb,
                          project="pcam-ablations",
                          tags={"ablation": cfg.name}) as logger:
        train(model, train_loader, val_loader, train_cfg, checkpoint_path=ckpt,
              on_epoch_end=lambda ep, m: logger.log_metrics(
                  {k: v for k, v in m.items() if isinstance(v, (int, float))}, step=ep))
        # Reload best checkpoint before test-set evaluation.
        state = torch.load(ckpt, map_location=train_cfg.device)
        model.load_state_dict(state["model_state_dict"])
        test_metrics = evaluate_and_save(model, test_loader, run_name, output_dir / "figures",
                                         device=train_cfg.device)
        logger.log_metrics({f"test_{k}": v for k, v in test_metrics.to_dict().items()})
        logger.log_artifact(ckpt)

    return {**cfg.__dict__, "seed": seed, **test_metrics.to_dict()}


def write_ablation_report(results: list[dict], output_path: Path, varying_key: str) -> None:
    """Write a Markdown comparison table for an ablation."""
    if not results:
        return
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "nll"]
    header = f"| {varying_key} | seed | " + " | ".join(metric_keys) + " |"
    sep = "|" + "---|" * (len(metric_keys) + 2)
    lines = [f"# Ablation: varying `{varying_key}`\n", header, sep]
    for r in results:
        row = f"| {r[varying_key]} | {r['seed']} | " + " | ".join(
            f"{r[k]:.4f}" for k in metric_keys
        ) + " |"
        lines.append(row)
    output_path.write_text("\n".join(lines))
    output_path.with_suffix(".json").write_text(json.dumps(results, indent=2))
