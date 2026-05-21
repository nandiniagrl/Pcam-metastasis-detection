# Reproduction guide

This document gives the **exact commands**, environment versions, and
expected outputs needed to reproduce every figure and table in this
repository.

## 1. Environment

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 (also tested on macOS 14, Windows 11) |
| Python | 3.10 |
| CUDA (GPU runs) | 11.8 or 12.1 |
| PyTorch | 2.3.1 |
| torchvision | 0.18.1 |
| scikit-learn | 1.5.1 |
| numpy | 1.26.4 |

Hardware used for the reported numbers: a single NVIDIA RTX-class GPU
(≥ 8 GB VRAM) is sufficient. CPU-only training of ResNet50 on the full
dataset is impractical (~ days); use `--subset-fraction 0.1` for CPU smoke
tests.

### Create the environment

```bash
# Option A — conda
conda env create -f environment.yml
conda activate pcam

# Option B — pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset

Download the six PCam `.h5` files from
<https://github.com/basveeling/pcam> and place them under `data/raw/`:

```
data/raw/
├── camelyonpatch_level_2_split_train_x.h5
├── camelyonpatch_level_2_split_train_y.h5
├── camelyonpatch_level_2_split_valid_x.h5
├── camelyonpatch_level_2_split_valid_y.h5
├── camelyonpatch_level_2_split_test_x.h5
└── camelyonpatch_level_2_split_test_y.h5
```

Verify:

```bash
python -m src.data.check_dataset --data-root data/raw
```

Expected output (sizes and class balance):

```
[train] N = 262,144 | class balance = {0: 131072, 1: 131072} (50.0% positive)
[valid] N = 32,768  | class balance = {0: 16384,  1: 16384}  (50.0% positive)
[test]  N = 32,768  | class balance = {0: 16384,  1: 16384}  (50.0% positive)
Saved sample grid → results/figures/sample_patches.png
```

## 3. Seeds used

All reported numbers use these three seeds:

```python
SEEDS = [42, 123, 2024]
```

Seeds are fixed in:

- `src/utils/seed.py::set_seed` — covers `random`, `numpy`, `torch` (CPU &
  CUDA), and `PYTHONHASHSEED`.
- `DataLoader` workers — via `seed_worker` and a per-run `torch.Generator`.
- cuDNN — `deterministic = True`, `benchmark = False`.

## 4. Headline results — run order

```bash
# step 1 — multi-seed training & evaluation of all three models
bash scripts/run_multi_seed.sh
# → produces results/{logistic_regression,custom_cnn,resnet50}_test_metrics_seed{42,123,2024}.json
# → produces results/model_comparison.{md,csv}
# → produces results/figures/model_comparison.png

# step 2 — three ablation studies, three seeds each (≈ 9 × 3 = 27 short runs)
bash scripts/run_all_ablations.sh
# → produces results/ablations/{augmentation,learning_rate,freeze}/report.md

# step 3 — D4 robustness on the best ResNet50 checkpoint
python -m src.evaluation.rotation_robustness \
    --checkpoint results/resnet50_seed42.pt --arch resnet50
# → results/rotation_robustness_summary.csv
# → results/figures/rotation_robustness_plot.png

# step 4 — Grad-CAM explainability
python -m src.explainability.gradcam_resnet50 \
    --checkpoint results/resnet50_seed42.pt
# → results/figures/resnet50_gradcam_examples.png
```

## 5. Inspecting runs

Every training run is logged to MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# then open http://127.0.0.1:5000
```

Pass `--wandb` to any training command to additionally log to
[Weights & Biases](https://wandb.ai). You need to be logged in
(`wandb login`).

## 6. Compute budget

Approximate wall-clock on a single RTX 3060:

| Step | Time |
|---|---|
| Logistic regression (×3 seeds) | ≈ 4 min |
| Custom CNN, 1 seed, 15 epochs | ≈ 25 min |
| ResNet50, 1 seed, 10 epochs (last_block) | ≈ 50 min |
| All three ablations (subset_fraction=0.2) | ≈ 2 h |
| D4 robustness eval | ≈ 4 min |
| Grad-CAM (8 patches) | < 1 min |
| **Total (single seed pass + ablations)** | **≈ 4 h** |
| **Total (multi-seed × 3 + ablations)** | **≈ 8 h** |

On CPU, only the LR baseline and Grad-CAM are practical.

## 7. Trouble-shooting

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError: PCam files for split 'train' not found` | `.h5` files not under `data/raw/` |
| OOM during ResNet50 | Reduce `training.batch_size` from 128 to 64 or 32 |
| Slow CPU run | Use `data.subset_fraction=0.1` for smoke test |
| Non-deterministic results across runs | Verify `set_seed` is called *before* dataset & dataloader creation |
| `wandb` blocks at start | Run `wandb offline` or omit `--wandb` |
