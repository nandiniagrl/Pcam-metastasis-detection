"""Rotation & reflection robustness evaluation.

Histopathology images have no canonical orientation (Veeling et al., 2018), so
a clinically reliable model must be approximately invariant to 90° rotations
and reflections. This script evaluates a trained model under all 8 elements
of the dihedral group D4 (4 rotations × 2 flips) and reports the spread.

Run with ``python -m src.evaluation.rotation_robustness --checkpoint ...``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import PCamDataset
from src.data.transforms import build_eval_transform
from src.models.custom_cnn import CustomCNN
from src.models.resnet50 import ResNet50Classifier
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed


TRANSFORMS = [
    ("identity",      lambda t: t),
    ("rot90",         lambda t: torch.rot90(t, 1, dims=(-2, -1))),
    ("rot180",        lambda t: torch.rot90(t, 2, dims=(-2, -1))),
    ("rot270",        lambda t: torch.rot90(t, 3, dims=(-2, -1))),
    ("flip_h",        lambda t: torch.flip(t, dims=(-1,))),
    ("flip_v",        lambda t: torch.flip(t, dims=(-2,))),
    ("flip_h_rot90",  lambda t: torch.rot90(torch.flip(t, dims=(-1,)), 1, dims=(-2, -1))),
    ("flip_v_rot90",  lambda t: torch.rot90(torch.flip(t, dims=(-2,)), 1, dims=(-2, -1))),
]


@torch.no_grad()
def evaluate_under_transform(model: nn.Module, loader: DataLoader, transform_fn, device: str
                             ) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    all_probs, all_labels = [], []
    for x, y in loader:
        x = transform_fn(x.to(device, non_blocking=True))
        probs = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def _build_model(arch: str, freeze: str) -> nn.Module:
    if arch == "custom_cnn":
        return CustomCNN()
    if arch == "resnet50":
        return ResNet50Classifier(freeze=freeze, pretrained=False)
    raise ValueError(f"Unknown architecture: {arch!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotation robustness evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", choices=["custom_cnn", "resnet50"], required=True)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze", default="last_block")
    parser.add_argument("--output-csv", default="results/rotation_robustness_summary.csv")
    parser.add_argument("--output-fig", default="results/figures/rotation_robustness_plot.png")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = PCamDataset(args.data_root, "test", build_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = _build_model(args.arch, args.freeze)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rows = []
    for name, fn in tqdm(TRANSFORMS, desc="D4 transforms"):
        probs, labels = evaluate_under_transform(model, test_loader, fn, device)
        m = compute_metrics(labels, probs)
        rows.append({"transform": name, **m.to_dict()})

    # Save CSV
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    csv_lines = [",".join(keys)] + [",".join(str(r[k]) for k in keys) for r in rows]
    Path(args.output_csv).write_text("\n".join(csv_lines))

    # Plot
    names = [r["transform"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    aucs = [r["roc_auc"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.4
    x = np.arange(len(names))
    ax.bar(x - width/2, accs, width, label="Accuracy")
    ax.bar(x + width/2, aucs, width, label="ROC-AUC")
    ax.axhline(np.mean(accs), color="C0", linestyle="--", alpha=0.5, label=f"mean acc = {np.mean(accs):.4f}")
    ax.axhline(np.mean(aucs), color="C1", linestyle="--", alpha=0.5, label=f"mean auc = {np.mean(aucs):.4f}")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25)
    ax.set_ylim(min(min(accs), min(aucs)) - 0.05, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"{args.arch}: D4-symmetry robustness on PCam test set")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(args.output_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_fig, dpi=150, bbox_inches="tight")

    summary = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)),
        "accuracy_range": float(max(accs) - min(accs)),
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs, ddof=1)),
        "roc_auc_range": float(max(aucs) - min(aucs)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
