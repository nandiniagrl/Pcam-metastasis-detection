# Contributing

## Development setup

```bash
git clone https://github.com/USERNAME/pcam-metastasis-detection.git
cd pcam-metastasis-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/   # should be green
```

## Style

- Type hints on new public functions.
- Docstrings in Google style (one-line summary, args, returns).
- Keep lines under ≈ 100 chars; tools like `ruff` or `black` are welcome.
- Don't put hyperparameters in source code — add them to a YAML in
  `configs/`.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add support for ResNet101 backbone
fix: handle zero-division in precision when threshold is too high
docs: clarify multi-seed reproduction recipe
test: add smoke test for Grad-CAM hook lifecycle
refactor(trainer): extract scheduler builder
chore: pin numpy to 1.26.4
```

Squash trivially-related commits before opening a PR.

## Branching

- `main` is the protected default branch — CI must pass.
- Topic branches: `feat/<short-name>`, `fix/<short-name>`,
  `docs/<short-name>`.

## Pull requests

A good PR:

1. Describes **what** changed and **why** (link the issue / rubric item).
2. Adds or updates tests for behavioural changes.
3. Updates relevant docs (README, REPRODUCE, REPORT).
4. Keeps the diff focused — separate refactors from new features.

## Reporting issues

Please include:

- the command you ran,
- the full traceback,
- your Python / PyTorch / CUDA versions,
- the config file or overrides used.
