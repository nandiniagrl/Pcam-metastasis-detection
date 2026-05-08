"""Inspect PCam: shapes, class balance, sample patches.

Run with: ``python -m src.data.check_dataset``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import PCamDataset
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the PCam dataset.")
    parser.add_argument("--data-root", default="data/raw", help="Directory with PCam .h5 files.")
    parser.add_argument("--output", default="results/figures/sample_patches.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        ds = PCamDataset(root=args.data_root, split=split, transform=None)
        balance = ds.class_balance()
        total = sum(balance.values())
        print(f"[{split}] N = {len(ds):,} | class balance = {balance} "
              f"({balance.get(1, 0) / total:.1%} positive)")

    # Visual sample grid from train split.
    train = PCamDataset(root=args.data_root, split="train", transform=None)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(train), size=16)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, idx in zip(axes.flat, indices, strict=True):
        img, label = train[int(idx)]
        ax.imshow(np.asarray(img))
        ax.set_title(f"label={int(label)}", fontsize=9)
        ax.axis("off")
    fig.suptitle("PCam sample patches (train split)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved sample grid → {out_path}")


if __name__ == "__main__":
    main()
