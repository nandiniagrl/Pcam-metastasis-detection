"""Aggregate per-seed metrics into a single comparison table.

Reads every ``*_test_metrics_seed*.json`` file under ``results/`` and produces
a Markdown table with mean ± std across seeds, plus a side-by-side bar chart.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.metrics import ClassificationMetrics, summarize_runs

SEED_PATTERN = re.compile(r"^(?P<model>[a-zA-Z0-9_]+?)_test_metrics_seed(?P<seed>\d+)\.json$")


def _load_metrics(path: Path) -> ClassificationMetrics:
    with path.open() as f:
        d = json.load(f)
    return ClassificationMetrics(**d)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models across seeds.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-md", default="results/model_comparison.md")
    parser.add_argument("--output-csv", default="results/model_comparison.csv")
    parser.add_argument("--output-fig", default="results/figures/model_comparison.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    by_model: dict[str, list[ClassificationMetrics]] = defaultdict(list)
    for f in sorted(results_dir.glob("*_test_metrics_seed*.json")):
        m = SEED_PATTERN.match(f.name)
        if not m:
            continue
        by_model[m["model"]].append(_load_metrics(f))

    # Fallback: a single-seed file like results/resnet50_test_metrics.json
    for f in sorted(results_dir.glob("*_test_metrics.json")):
        name = f.stem.replace("_test_metrics", "")
        if name not in by_model:
            by_model[name].append(_load_metrics(f))

    if not by_model:
        raise SystemExit(
            "No metric files found. Train and evaluate at least one model first."
        )

    # Markdown table.
    md_lines = ["# Model comparison\n",
                "Mean ± std over multiple seeds. Single-seed entries report only the value.\n",
                "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | NLL | # seeds |",
                "|---|---|---|---|---|---|---|---|"]
    csv_lines = ["model,n_seeds,accuracy_mean,accuracy_std,precision_mean,precision_std,"
                 "recall_mean,recall_std,f1_mean,f1_std,roc_auc_mean,roc_auc_std,nll_mean,nll_std"]

    fig_data: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, runs in by_model.items():
        summary = summarize_runs(runs)
        fig_data[model_name] = summary
        n = len(runs)

        def fmt(field: str) -> str:
            s = summary[field]
            return f"{s['mean']:.4f} ± {s['std']:.4f}" if n > 1 else f"{s['mean']:.4f}"

        md_lines.append(
            f"| {model_name} | {fmt('accuracy')} | {fmt('precision')} | {fmt('recall')} | "
            f"{fmt('f1')} | {fmt('roc_auc')} | {fmt('nll')} | {n} |"
        )
        csv_lines.append(
            f"{model_name},{n}," + ",".join(
                f"{summary[f]['mean']:.6f},{summary[f]['std']:.6f}"
                for f in ("accuracy", "precision", "recall", "f1", "roc_auc", "nll")
            )
        )

    Path(args.output_md).write_text("\n".join(md_lines))
    Path(args.output_csv).write_text("\n".join(csv_lines))
    print(f"Wrote {args.output_md} and {args.output_csv}")

    # Bar chart with error bars.
    models = list(fig_data.keys())
    metrics_to_plot = ["accuracy", "f1", "roc_auc"]
    width = 0.25
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(models)), 5))
    for i, metric in enumerate(metrics_to_plot):
        means = [fig_data[m][metric]["mean"] for m in models]
        stds = [fig_data[m][metric]["std"] for m in models]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4, label=metric)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("PCam test-set performance (multi-seed)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(args.output_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_fig, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output_fig}")


if __name__ == "__main__":
    main()
