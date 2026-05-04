"""Canonical metric computation for binary classification.

Centralising metric calculation here ensures the LR baseline, custom CNN, and
ResNet50 are all evaluated identically — a key reproducibility consideration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Container for a complete binary-classification evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    nll: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute the full metric suite for a single evaluation.

    Args:
        y_true: 1-D array of true labels in {0, 1}.
        y_prob: 1-D array of predicted positive-class probabilities in [0, 1].
        threshold: Decision threshold for hard labels.

    Returns:
        ClassificationMetrics with all canonical metrics.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    # log_loss is unstable when probs are exactly 0 or 1
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob),
        nll=log_loss(y_true, y_prob_clipped),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def summarize_runs(metrics_list: list[ClassificationMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate metrics across multiple runs (different seeds).

    Returns ``{metric_name: {"mean": ..., "std": ..., "min": ..., "max": ...}}``.
    """
    if not metrics_list:
        return {}

    fields = ["accuracy", "precision", "recall", "f1", "roc_auc", "nll"]
    summary: dict[str, dict[str, float]] = {}
    for f in fields:
        vals = np.array([getattr(m, f) for m in metrics_list], dtype=float)
        summary[f] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return summary
