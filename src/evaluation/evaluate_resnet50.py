"""Evaluate a trained ResNet50 on the PCam test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PCamDataset
from src.data.transforms import build_eval_transform
from src.evaluation.evaluate import evaluate_and_save
from src.models.resnet50 import ResNet50Classifier
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ResNet50.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze", default="last_block")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = PCamDataset(args.data_root, "test", build_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = ResNet50Classifier(freeze=args.freeze, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_and_save(model, test_loader, "resnet50", Path(args.output_dir), device=device)
    print(json.dumps(metrics.to_dict(), indent=2))

    out_json = Path("results") / "resnet50_test_metrics.json"
    out_json.write_text(json.dumps(metrics.to_dict(), indent=2))
    print(f"Saved metrics → {out_json}")


if __name__ == "__main__":
    main()
