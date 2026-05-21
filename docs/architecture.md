# Architecture & design notes

## Why this layout

The codebase is split into orthogonal modules so that any single change
(adding a model, swapping a metric, running a new ablation) touches the
smallest possible surface:

```
src/
├── data/           # what the input looks like
├── models/         # what we run on it
├── training/       # how we fit the models  (only ONE training loop)
├── evaluation/     # how we measure them    (only ONE metrics module)
├── explainability/ # how we explain them
├── ablations/      # how we vary them
└── utils/          # cross-cutting concerns: seeding, logging, config
```

**Single source of truth for metrics.** `src/utils/metrics.py` is the only
place numbers come from. The LR script, the CNN evaluator, the ResNet50
evaluator, and every ablation all call the same `compute_metrics(y_true,
y_prob)` function. This kills entire classes of "model A and model B aren't
strictly comparable" bugs.

**Single training loop.** `src/training/trainer.py::train` is reused by
every PyTorch model and every ablation. Adding a new architecture requires
only a new file under `src/models/`; the training behaviour is identical by
construction.

## Reproducibility design

- `set_seed(seed)` is the first call in every entry point and seeds Python,
  NumPy, PyTorch (CPU + CUDA), `PYTHONHASHSEED`, and cuDNN.
- `DataLoader`s receive a `torch.Generator` seeded per run and a
  `worker_init_fn` so each worker is independently seeded.
- All hyperparameters live in YAML under `configs/`, never in source.
- The dependency lock files (`requirements.txt`, `environment.yml`) pin
  every relevant package to a known-good version.

## Experiment tracking strategy

Two backends, used together where possible:

| Backend | Purpose |
|---|---|
| **MLflow** (always on) | Local, file-backed source of truth — no internet required, runs survive offline. |
| **Weights & Biases** (`--wandb` flag) | Cloud-hosted dashboards, easier sharing, better diffing across runs. |

The `ExperimentLogger` in `src/utils/logger.py` writes to both with one
call. Failure of one backend never affects the other or the training run
itself.

## Training-loop design choices

- **Early stopping on validation loss**, not on a metric. Val-loss is
  smoother and better correlated with calibration than accuracy.
- **`ReduceLROnPlateau`** by default. Cosine schedule is also supported via
  config — kept available for ablations.
- **Best-checkpoint saving** based on val-loss. We never evaluate the
  final-epoch weights on the test set; they are a noisy estimate.
- **Gradient clipping at norm 1.0.** Defensive against exploding gradients
  when fine-tuning pretrained weights with a high learning rate.

## Augmentation choice

Augmentations are chosen to **respect the symmetries of histopathology**
rather than to maximally distort images:

- D4 group (horizontal flip, vertical flip, 90° rotations) — no canonical
  orientation in tissue under microscope.
- Colour jitter — approximates inter-laboratory stain variation.
- **No random crops** — patches are already 96×96 and centred on the
  region of interest; cropping would remove the labelled centre.
- **No erasure / cutout** — could remove tumour pixels and silently flip
  the label.

## Where to extend

| To add a... | Touch these files |
|---|---|
| new model | `src/models/<name>.py`, plus a `train_<name>.py` and `evaluate_<name>.py` mirroring the existing scripts |
| new metric | `src/utils/metrics.py::compute_metrics` and `ClassificationMetrics` |
| new ablation | `src/ablations/ablation_<name>.py`, reusing `ablation_runner.run_single` |
| new explanation method | `src/explainability/<method>_<arch>.py` |
| new dataset | `src/data/<dataset>.py` + transforms; nothing else changes |
