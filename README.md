# PCam Metastasis Detection

[![CI](https://github.com/USERNAME/pcam-metastasis-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/pcam-metastasis-detection/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-blue.svg)](https://mlflow.org/)
[![Weights & Biases](https://img.shields.io/badge/tracking-W%26B-FFBE00.svg)](https://wandb.ai/)

Automated detection of breast-cancer metastases in lymph-node histopathology
patches, using the [PatchCamelyon (PCam)](https://github.com/basveeling/pcam)
dataset of 327,680 96×96 RGB patches derived from the
[CAMELYON16](https://camelyon16.grand-challenge.org/) challenge.

This project implements and compares three modelling approaches — a classical
**Logistic Regression** baseline, a **Custom CNN** deep-learning baseline, and
a transfer-learned **ResNet50** — with three controlled ablations, multi-seed
variance analysis, D4-symmetry robustness evaluation, and Grad-CAM
explainability.

> **Quick links:** &nbsp;
> [Reproduction guide](REPRODUCE.md) ·
> [Full report](reports/REPORT.md) ·
> [Architecture notes](docs/architecture.md)

---

## Table of contents

1. [Motivation & clinical context](#motivation--clinical-context)
2. [Dataset](#dataset)
3. [Learning task & evaluation protocol](#learning-task--evaluation-protocol)
4. [Methods & baselines](#methods--baselines)
5. [Ablation studies](#ablation-studies)
6. [Robustness & reproducibility](#robustness--reproducibility)
7. [Results](#results)
8. [Limitations, ethics & clinical relevance](#limitations-ethics--clinical-relevance)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Repository structure](#repository-structure)
12. [References](#references)

---

## Motivation & clinical context

Metastasis to sentinel lymph nodes is one of the strongest prognostic
indicators in breast cancer staging, and current diagnostic practice requires
a pathologist to manually examine gigapixel whole-slide images for small
clusters of tumour cells. The process is time-consuming, exhibits
inter-observer variability, and is prone to fatigue-induced misses
(Bejnordi et al., 2017).

Patch-based deep learning has been shown to match or exceed expert
pathologists on this task (Liu et al., 2017; Veeling et al., 2018). The
**PatchCamelyon** dataset enables principled, reproducible benchmarking of
candidate models without requiring whole-slide-image infrastructure: each
patch is independently labelled as tumour-containing or not, transforming the
clinical workflow into a clean binary-classification problem suitable as a
testbed for medical-image ML research.

This project frames the problem as **AI-assisted clinical decision support**:
the goal is not to replace the pathologist, but to flag suspicious regions
with calibrated confidence and interpretable evidence (Grad-CAM heatmaps),
reducing time-to-diagnosis and missed cases.

## Dataset

| Property | Value |
|---|---|
| Source | [PCam (Veeling et al., 2018)](https://github.com/basveeling/pcam), derived from CAMELYON16 |
| Total patches | 327,680 |
| Patch size | 96 × 96 RGB |
| Magnification | 10× (0.972 µm/pixel) |
| Splits | 262,144 train / 32,768 valid / 32,768 test (75/12.5/12.5) |
| Labels | Binary — positive iff the central 32×32 region contains tumour tissue |
| Selection | Patches chosen via hard-negative mining from CAMELYON16 |
| Class balance | Approximately 50/50 — no resampling needed |

**Setup.** Download the six `.h5` files from the official PCam repository and
place them under `data/raw/`. See [REPRODUCE.md](REPRODUCE.md) for details.

A visual sanity check is provided:

```bash
python -m src.data.check_dataset --data-root data/raw
# → prints split sizes & class balance
# → writes results/figures/sample_patches.png
```

## Learning task & evaluation protocol

**Task.** Supervised binary classification: given a 96×96 RGB patch, predict
the probability that it contains tumour tissue.

**Loss.** Cross-entropy with a softmax head (equivalent to binary log-loss).

**Evaluation metrics** (all computed in `src/utils/metrics.py` so every model
is evaluated identically):

| Metric | Why it matters |
|---|---|
| Accuracy | Headline figure on a balanced dataset |
| Precision, Recall, F1 | Class-specific performance — false negatives are clinically expensive |
| ROC-AUC | Threshold-independent ranking quality |
| Negative log-likelihood (NLL) | Calibration — a confident wrong model has high NLL |
| Confusion matrix | Where the model fails |

**Qualitative evaluation** complements the metric table:

- **Grad-CAM** heatmaps (Selvaraju et al., 2017) on the final ResNet50 layer,
  visualising the regions that drive each prediction.
- **D4-symmetry robustness**: the same model is evaluated under all 8
  rotation/reflection variants of the test set, and the spread quantifies how
  much its decisions depend on input orientation — a clinically meaningful
  property given histopathology has no canonical orientation
  (Veeling et al., 2018).

**Statistical rigour.** Every neural model is trained with three random seeds
(42, 123, 2024) and we report **mean ± standard deviation** of each test-set
metric. Aggregation is implemented in
[`src/evaluation/compare_models.py`](src/evaluation/compare_models.py).

## Methods & baselines

We satisfy the requirement of "at least two baselines and two modelling
approaches" by implementing three distinct model families:

### Baseline 1 — Logistic Regression on pixel features (classical)

Patches are flattened to 27,648-dimensional vectors, standardised, and fed
into an L2-regularised logistic regression (`saga` solver). This establishes
a non-deep-learning floor: if a deep model cannot meaningfully beat raw-pixel
logistic regression, something is wrong with training.

### Baseline 2 — Custom CNN (deep-learning baseline, no transfer)

Four convolutional blocks (32→64→128→256 channels) with BatchNorm + ReLU +
max-pool, followed by a small fully-connected head with dropout. Trained
from scratch. Tests whether the dataset is large enough to learn useful
representations end-to-end.

### Main model — ResNet50 with transfer learning (final approach)

ImageNet-pretrained ResNet50 (`IMAGENET1K_V2` weights) with the final
fully-connected layer replaced by a 2-class head. Default freezing strategy:
`last_block` — only `layer4` and the head are trainable, balancing
representation quality (from ImageNet) with task-specific adaptation. This
choice is empirically justified in the [freezing ablation](#ablation-studies).

ResNet50 is suitable for **Grad-CAM** because its final convolutional block
(`layer4`) has 2048 channels at 3×3 spatial resolution — fine enough for
localisation, coarse enough for meaningful semantic content.

### Note on the paper's P4M-DenseNet

Veeling et al. (2018) propose a group-equivariant DenseNet that exploits the
inherent rotation/reflection symmetry of histopathology images and reports
state-of-the-art results on PCam (96.3% AUC). We **do not** reimplement that
architecture here (it requires a specialised G-CNN framework), but we
explicitly evaluate D4 symmetry robustness on our ResNet50 to quantify the
gap that a fully equivariant model would close. This is discussed in
[REPORT.md](reports/REPORT.md).

## Ablation studies

We perform **three independent ablations**, each varying one design
choice while holding the others fixed, with **three seeds per setting** for
variance estimation. All ablation runs are logged to MLflow under the
`pcam-ablations` experiment (and optionally W&B).

| # | Ablation | Varies | What it isolates |
|---|---|---|---|
| 1 | Data augmentation | `augment ∈ {on, off}` | Effect of D4 + colour-jitter augmentation, which directly tests Veeling et al.'s rotation-equivariance argument |
| 2 | Learning rate | `lr ∈ {1e-3, 1e-4, 1e-5}` | Sensitivity to optimisation hyperparameters; informs the default config |
| 3 | Freezing strategy | `freeze ∈ {all_but_fc, last_block, none}` | How much transferred ImageNet features need to be adapted vs. reused |

Each ablation script ([`ablation_augmentation.py`](src/ablations/ablation_augmentation.py),
[`ablation_learning_rate.py`](src/ablations/ablation_learning_rate.py),
[`ablation_freeze.py`](src/ablations/ablation_freeze.py)) writes a Markdown
report and JSON record under `results/ablations/<name>/`.

Run them all:

```bash
bash scripts/run_all_ablations.sh
```

## Robustness & reproducibility

**Variance analysis.** All metric tables report mean ± std across three
seeds. The default seeds are `{42, 123, 2024}` (chosen before any results
were observed).

**D4 robustness.** `python -m src.evaluation.rotation_robustness` evaluates
the trained ResNet50 under all 8 elements of the dihedral group D4 (4
rotations × 2 flips). It writes a CSV of per-transform metrics and a bar
chart; the spread (std and range across transforms) quantifies orientation
robustness. Veeling et al. (2018) show that a non-equivariant DenseNet
exhibits noticeable fluctuations under rotation, so we expect ResNet50 +
augmentation to be reasonable but imperfect on this test.

**Reproducibility checklist.**

- [x] Fixed seeds in `src/utils/seed.py` (`set_seed`) covering Python,
      NumPy, PyTorch (CPU + CUDA), and `PYTHONHASHSEED`.
- [x] Deterministic cuDNN mode (`torch.backends.cudnn.deterministic = True`)
      enabled by default.
- [x] Seeded `DataLoader` workers via `seed_worker` and a per-run
      `torch.Generator`.
- [x] Pinned dependency versions in [`requirements.txt`](requirements.txt)
      and [`environment.yml`](environment.yml).
- [x] All hyperparameters in version-controlled YAML
      ([`configs/`](configs/)).
- [x] Step-by-step reproduction script in [`REPRODUCE.md`](REPRODUCE.md).
- [x] CI workflow (GitHub Actions) runs unit tests on every push.
- [x] Experiments logged to MLflow (default) and optionally Weights & Biases.

## Results

Results are produced by running:

```bash
bash scripts/run_multi_seed.sh           # train + evaluate all three models, all seeds
python -m src.evaluation.compare_models  # aggregate
```

This writes:

- `results/model_comparison.md` — mean ± std table over three seeds
- `results/model_comparison.csv` — machine-readable version
- `results/figures/model_comparison.png` — bar chart with error bars

A detailed walkthrough of results, Grad-CAM examples, robustness curves, and
discussion of failure modes is in [`reports/REPORT.md`](reports/REPORT.md).

## Limitations, ethics & clinical relevance

**Limitations of patch-level modelling.**

- PCam patches are 96×96 at 10× magnification — far smaller than the tissue
  context a pathologist routinely uses. A real diagnostic system would
  aggregate predictions across an entire whole-slide image (WSI).
- The dataset was curated by hard-negative mining, which makes it harder than
  uniform random sampling but **not representative of WSI prevalence**
  (positives are rare in real screening).
- Inter-laboratory stain variation, scanner-induced colour shifts, and tissue
  preparation artefacts are only partially captured by `ColorJitter`
  augmentation. External validation on a held-out laboratory is required
  before any clinical claim.

**Domain shift & generalisation.** A model trained on CAMELYON16-derived
patches may underperform on slides from a different scanner or staining
protocol. This is well-documented in computational pathology and is a major
deployment risk (Bejnordi et al., 2016; Tellez et al., 2019).

**Ethical considerations.**

- The model produces a probability, not a diagnosis. Calibration errors at
  the decision threshold translate directly into false negatives (missed
  cancers) or false positives (unnecessary anxiety and follow-up). Any
  clinical deployment must include calibration analysis on the deployment
  population.
- Algorithmic decision support carries **automation-bias risk**: a confident
  AI output can dissuade a pathologist from a closer look. Grad-CAM
  visualisations partly mitigate this by surfacing model reasoning, but
  cannot eliminate the bias.
- PCam contains no patient identifiers but is derived from real patients.
  Re-identification is implausible at the patch level but must be considered
  if WSIs are reintegrated.

**Intended use.** The system is intended only as an assistive triage and
visualisation tool, never as an autonomous diagnostic device. It is not a
substitute for expert pathologist judgement and has not been validated
clinically.

## Installation

### Option A — conda (recommended for full reproducibility)

```bash
git clone https://github.com/USERNAME/pcam-metastasis-detection.git
cd pcam-metastasis-detection
conda env create -f environment.yml
conda activate pcam
```

### Option B — pip + venv

```bash
git clone https://github.com/USERNAME/pcam-metastasis-detection.git
cd pcam-metastasis-detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

GPU users should install the matching CUDA build of PyTorch — see
[pytorch.org](https://pytorch.org/get-started/locally/).

### Dataset

Run `bash scripts/download_data.sh` for instructions. The six PCam `.h5`
files must end up in `data/raw/`.

## Usage

```bash
# 1. sanity-check the data
python -m src.data.check_dataset

# 2. train the three models (single seed)
python -m src.models.logistic_regression --seed 42
python -m src.training.train_custom_cnn --seed 42
python -m src.training.train_resnet50 --seed 42

# 3. evaluate
python -m src.evaluation.evaluate_custom_cnn --checkpoint results/custom_cnn_seed42.pt
python -m src.evaluation.evaluate_resnet50 --checkpoint results/resnet50_seed42.pt

# 4. multi-seed runs + comparison table
bash scripts/run_multi_seed.sh

# 5. ablations
bash scripts/run_all_ablations.sh

# 6. robustness analysis
python -m src.evaluation.rotation_robustness \
    --checkpoint results/resnet50_seed42.pt --arch resnet50

# 7. Grad-CAM
python -m src.explainability.gradcam_resnet50 \
    --checkpoint results/resnet50_seed42.pt

# 8. inspect runs in MLflow
mlflow ui  # then visit http://127.0.0.1:5000
```

Hyperparameters live in [`configs/`](configs/) and can be overridden from the
command line, e.g.:

```bash
python -m src.training.train_resnet50 training.lr=5e-5 training.epochs=8
```

## Repository structure

```
pcam-metastasis-detection/
├── configs/                  # YAML hyperparameter configs (one per model)
├── data/
│   ├── raw/                  # ← place .h5 files here (gitignored)
│   └── processed/
├── docs/                     # architecture notes & design docs
├── reports/
│   └── REPORT.md             # full results write-up
├── results/                  # metrics, figures, ablation outputs (mostly gitignored)
├── scripts/                  # bash drivers (multi-seed, ablations, data download)
├── src/
│   ├── data/                 # dataset class, transforms, sanity-check CLI
│   ├── models/               # LR, custom CNN, ResNet50
│   ├── training/             # generic trainer + per-model entry points
│   ├── evaluation/           # eval, comparison, rotation robustness
│   ├── explainability/       # Grad-CAM
│   ├── ablations/            # three ablation experiments
│   └── utils/                # seeding, logging, metrics, config loading
├── tests/                    # pytest unit tests
├── .github/workflows/ci.yml  # CI: tests on every push
├── environment.yml           # conda env
├── requirements.txt          # pip deps (pinned)
├── REPRODUCE.md              # exact reproduction recipe
├── LICENSE                   # MIT
└── README.md                 # this file
```

## References

1. Veeling, B.S., Linmans, J., Winkens, J., Cohen, T., Welling, M. (2018).
   *Rotation Equivariant CNNs for Digital Pathology*. MICCAI.
2. Selvaraju, R.R. et al. (2017). *Grad-CAM: Visual Explanations from Deep
   Networks via Gradient-based Localization*. ICCV.
3. Bejnordi, B.E. et al. (2017). *Diagnostic Assessment of Deep Learning
   Algorithms for Detection of Lymph Node Metastases in Women With Breast
   Cancer*. JAMA.
4. Liu, Y. et al. (2017). *Detecting Cancer Metastases on Gigapixel
   Pathology Images*. arXiv:1703.02442.
5. He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep Residual Learning for
   Image Recognition*. CVPR.
6. Tellez, D. et al. (2019). *Quantifying the effects of data augmentation
   and stain color normalization in convolutional neural networks for
   computational pathology*. Medical Image Analysis.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

The PCam dataset is distributed by Veeling et al. under
[CC0 1.0](https://github.com/basveeling/pcam) and is **not** redistributed
here.
