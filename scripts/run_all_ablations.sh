#!/usr/bin/env bash
# Run all three ablation studies in sequence.
# Usage: bash scripts/run_all_ablations.sh [--wandb]
set -euo pipefail

WANDB_FLAG=""
if [[ "${1:-}" == "--wandb" ]]; then
  WANDB_FLAG="--wandb"
fi

echo "=== Ablation 1/3: data augmentation ==="
python -m src.ablations.ablation_augmentation $WANDB_FLAG

echo "=== Ablation 2/3: learning rate ==="
python -m src.ablations.ablation_learning_rate $WANDB_FLAG

echo "=== Ablation 3/3: freezing strategy ==="
python -m src.ablations.ablation_freeze $WANDB_FLAG

echo "All ablations complete. Reports under results/ablations/*/report.md"
