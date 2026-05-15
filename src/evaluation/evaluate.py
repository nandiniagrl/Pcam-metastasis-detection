"""Generic evaluation utilities — loaded checkpoint + dataloader → metrics + figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.metrics import ClassificationMetrics, compute_metrics


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities for class 1, true labels) for the entire loader."""
    model.eval().to(device)
    all_probs, all_labels = [], []
    for x, y in tqdm(loader, desc="predict", leave=False):
        x = x.to(device, non_blocking=True)
        probs = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def plot_confusion_matrix(metrics: ClassificationMetrics, output_path: Path, title: str) -> None:
    cm = np.array([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "tumor"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = float(np.trapz(tpr, fpr))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_save(
    model: nn.Module,
    test_loader: DataLoader,
    name: str,
    output_dir: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ClassificationMetrics:
    """Run prediction, compute metrics, and save confusion matrix + ROC curve."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probs, labels = predict(model, test_loader, device)
    metrics = compute_metrics(labels, probs)

    plot_confusion_matrix(metrics, output_dir / f"{name}_confusion_matrix.png", f"{name} — confusion matrix")
    plot_roc(labels, probs, output_dir / f"{name}_roc_curve.png", f"{name} — ROC curve")

    np.savez(output_dir / f"{name}_predictions.npz", probs=probs, labels=labels)
    return metrics
