"""Ablation 2: effect of learning rate.

Sweeps learning rate over {1e-3, 1e-4, 1e-5}, keeping everything else fixed,
with multiple seeds per setting to estimate variance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ablations.ablation_runner import AblationConfig, run_single, write_ablation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: learning rate.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--subset-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="results/ablations/learning_rate")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    results = []
    for lr in args.lrs:
        for seed in args.seeds:
            cfg = AblationConfig(name=f"lr-{lr:.0e}", lr=lr, epochs=args.epochs)
            r = run_single(cfg, args.data_root, seed, out,
                           subset_fraction=args.subset_fraction,
                           use_wandb=args.wandb)
            results.append(r)

    write_ablation_report(results, out / "report.md", varying_key="lr")
    print(f"Done. Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
