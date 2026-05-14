"""Train ResNet50 (transfer learning) on PCam.

Run with ``python -m src.training.train_resnet50 --seed 42``.
Multi-seed runs are performed by ``scripts/run_multi_seed.sh``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PCamDataset
from src.data.transforms import build_eval_transform, build_train_transform
from src.models.resnet50 import ResNet50Classifier
from src.training.trainer import TrainConfig, train
from src.utils.config import load_config, merge_overrides
from src.utils.logger import ExperimentLogger
from src.utils.seed import seed_worker, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet50 on PCam.")
    parser.add_argument("--config", default="configs/resnet50.yaml")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = merge_overrides(load_config(args.config), args.overrides)
    set_seed(args.seed)

    train_ds = PCamDataset(
        args.data_root, "train", build_train_transform(cfg.training.augment),
        subset_fraction=cfg.data.subset_fraction,
    )
    val_ds = PCamDataset(args.data_root, "valid", build_eval_transform())

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=True,
    )

    model = ResNet50Classifier(freeze=cfg.model.freeze, pretrained=cfg.model.pretrained)
    print(f"[resnet50] trainable parameters: {model.trainable_parameter_count():,}")

    train_cfg = TrainConfig(
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        optimizer=cfg.training.optimizer,
        scheduler=cfg.training.scheduler,
        patience=cfg.training.patience,
    )

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / f"resnet50_seed{args.seed}.pt"

    run_name = f"resnet50_seed{args.seed}_freeze-{cfg.model.freeze}"
    log_config = {
        "model": "resnet50",
        "seed": args.seed,
        **dict(cfg.training),
        **dict(cfg.model),
    }
    with ExperimentLogger(run_name, log_config, use_wandb=args.wandb) as logger:
        result = train(
            model, train_loader, val_loader, train_cfg,
            checkpoint_path=ckpt,
            on_epoch_end=lambda ep, m: logger.log_metrics(
                {k: v for k, v in m.items() if isinstance(v, (int, float))}, step=ep
            ),
        )
        logger.log_artifact(ckpt)

    history_path = output_dir / f"resnet50_seed{args.seed}_history.json"
    history_path.write_text(json.dumps(result, indent=2))
    print(f"Saved history → {history_path}")


if __name__ == "__main__":
    main()
