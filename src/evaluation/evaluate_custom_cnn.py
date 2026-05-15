"""Evaluate a trained Custom CNN on the PCam test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PCamDataset
from src.data.transforms import build_eval_transform
from src.evaluation.evaluate import evaluate_and_save
from src.models.custom_cnn import CustomCNN
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Custom CNN.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = PCamDataset(args.data_root, "test", build_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = CustomCNN()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_and_save(model, test_loader, "custom_cnn", Path(args.output_dir), device=device)
    print(json.dumps(metrics.to_dict(), indent=2))

    out_json = Path("results") / "custom_cnn_test_metrics.json"
    out_json.write_text(json.dumps(metrics.to_dict(), indent=2))
    print(f"Saved metrics → {out_json}")


if __name__ == "__main__":
    main()
