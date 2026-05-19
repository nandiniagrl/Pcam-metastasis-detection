"""Ablation 3: effect of freezing strategy for ResNet50.

Compares three transfer-learning regimes:
  - ``all_but_fc``: linear probe — backbone frozen, only the head trained.
  - ``last_block``: fine-tune ``layer4`` + head (default).
  - ``none``: full fine-tuning of all parameters.

This isolates how much of the gain comes from feature reuse vs. adaptation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ablations.ablation_runner import AblationConfig, run_single, write_ablation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: freeze strategy.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--strategies", nargs="+",
                        default=["all_but_fc", "last_block", "none"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--subset-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="results/ablations/freeze")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    results = []
    for strategy in args.strategies:
        for seed in args.seeds:
            cfg = AblationConfig(name=f"freeze-{strategy}", freeze=strategy, epochs=args.epochs)
            r = run_single(cfg, args.data_root, seed, out,
                           subset_fraction=args.subset_fraction,
                           use_wandb=args.wandb)
            results.append(r)

    write_ablation_report(results, out / "report.md", varying_key="freeze")
    print(f"Done. Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
