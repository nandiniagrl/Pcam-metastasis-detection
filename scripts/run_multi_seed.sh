#!/usr/bin/env bash
# Train each model across three seeds and aggregate metrics.
# Usage: bash scripts/run_multi_seed.sh [--wandb]
#
# Multiple-seed runs are required by the rubric to demonstrate variance
# analysis. After this script finishes, ``compare_models.py`` will produce
# a single Markdown table with mean ± std across all three seeds.
set -euo pipefail

SEEDS=(42 123 2024)
WANDB_FLAG=""
if [[ "${1:-}" == "--wandb" ]]; then
  WANDB_FLAG="--wandb"
fi

mkdir -p results

# Logistic regression — single fit per seed (it's cheap).
for s in "${SEEDS[@]}"; do
  echo "=== Logistic Regression — seed $s ==="
  python -m src.models.logistic_regression --seed "$s" --output-dir results
  mv results/logreg_metrics.json "results/logistic_regression_test_metrics_seed${s}.json"
done

# Custom CNN
for s in "${SEEDS[@]}"; do
  echo "=== Custom CNN — seed $s ==="
  python -m src.training.train_custom_cnn --seed "$s" --output-dir results $WANDB_FLAG
  python -m src.evaluation.evaluate_custom_cnn \
    --checkpoint "results/custom_cnn_seed${s}.pt" --seed "$s"
  mv results/custom_cnn_test_metrics.json "results/custom_cnn_test_metrics_seed${s}.json"
done

# ResNet50
for s in "${SEEDS[@]}"; do
  echo "=== ResNet50 — seed $s ==="
  python -m src.training.train_resnet50 --seed "$s" --output-dir results $WANDB_FLAG
  python -m src.evaluation.evaluate_resnet50 \
    --checkpoint "results/resnet50_seed${s}.pt" --seed "$s"
  mv results/resnet50_test_metrics.json "results/resnet50_test_metrics_seed${s}.json"
done

python -m src.evaluation.compare_models
echo "All done. See results/model_comparison.md"
