"""Logistic Regression baseline on flattened pixel features.

This is the classical baseline mandated by the methodology requirement.
Despite its simplicity, it provides a meaningful lower bound: anything below
LR performance on PCam indicates a training bug rather than a hard problem.

Run with: ``python -m src.models.logistic_regression``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tqdm import tqdm

import joblib  # type: ignore[import-untyped]
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.dataset import PCamDataset
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed

def _load_split_as_array(root: str, split: str, max_samples: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Load a split as a flat ``(N, 96*96*3)`` float array."""
    ds = PCamDataset(root=root, split=split, transform=None)
    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    x = np.zeros((n, 96 * 96 * 3), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in tqdm(range(n), desc=f"Loading {split}"):
        img, label = ds[i]
        x[i] = (np.asarray(img).astype(np.float32) / 255.0).reshape(-1)
        y[i] = int(label)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Logistic Regression baseline on PCam.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train",
        type=int,
        default=30_000,
        help="Subsample training set for tractability (LR on 262k×27k features "
        "is impractical in memory). 30k samples is a strong baseline.",
    )
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[LR] Loading training subset…")
    x_train, y_train = _load_split_as_array(args.data_root, "train", args.max_train)
    print(f"[LR] Train shape: {x_train.shape}  positive rate: {y_train.mean():.3f}")

    print("[LR] Loading test split…")
    x_test, y_test = _load_split_as_array(args.data_root, "test", args.max_eval)

    print("[LR] Fitting StandardScaler…")
    scaler = StandardScaler(with_mean=False)  # sparse-friendly; avoids huge centred matrix
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    print("[LR] Fitting LogisticRegression…")
    clf = LogisticRegression(
        solver="saga",
        max_iter=50,
        C=1.0,
        random_state=args.seed,
        verbose=1,
    )
    clf.fit(x_train_s, y_train)

    probs = clf.predict_proba(x_test_s)[:, 1]
    metrics = compute_metrics(y_test, probs)

    print("[LR] Test metrics:")
    for k, v in metrics.to_dict().items():
        print(f"    {k}: {v}")

    (out / "logreg_metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2))
    joblib.dump({"scaler": scaler, "model": clf}, out / "logreg_model.joblib")
    print(f"[LR] Saved metrics → {out / 'logreg_metrics.json'}")
    print(f"[LR] Saved model   → {out / 'logreg_model.joblib'}")


if __name__ == "__main__":
    main()
