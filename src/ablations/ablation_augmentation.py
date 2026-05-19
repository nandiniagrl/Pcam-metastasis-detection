"""Ablation 1: effect of data augmentation.

Compares ResNet50 with and without histopathology-appropriate augmentation
(D4 rotations/flips + mild colour jitter). The rotation-equivariance argument
from Veeling et al. (2018) predicts that augmentation matters more on smaller
data regimes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ablations.ablation_runner import AblationConfig, run_single, write_ablation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: data augmentation.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--subset-fraction", type=float, default=0.2,
                        help="Use a subset of training data so the ablation runs in reasonable time.")
    parser.add_argument("--output-dir", default="results/ablations/augmentation")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    results = []
    for augment in (False, True):
        for seed in args.seeds:
            cfg = AblationConfig(
                name=f"aug-{augment}",
                augment=augment,
                epochs=args.epochs,
            )
            r = run_single(cfg, args.data_root, seed, out,
                           subset_fraction=args.subset_fraction,
                           use_wandb=args.wandb)
            r["augmentation"] = augment
            results.append(r)

    write_ablation_report(results, out / "report.md", varying_key="augmentation")
    print(f"Done. Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
