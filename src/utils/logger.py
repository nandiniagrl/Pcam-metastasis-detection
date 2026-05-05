"""Unified experiment logging across MLflow and Weights & Biases.

Both backends are optional and gracefully degrade if not installed or not
configured. Use one or both depending on the experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False


class ExperimentLogger:
    """Logs hyperparameters, metrics, and artifacts to MLflow and/or wandb.

    Designed to fail silently if a backend is misconfigured — training never
    crashes because of logging.
    """

    def __init__(
        self,
        run_name: str,
        config: Mapping[str, Any],
        use_mlflow: bool = True,
        use_wandb: bool = False,
        project: str = "pcam-metastasis-detection",
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.run_name = run_name
        self.config = dict(config)
        self.use_mlflow = use_mlflow and _MLFLOW_AVAILABLE
        self.use_wandb = use_wandb and _WANDB_AVAILABLE
        self.project = project
        self.tags = dict(tags) if tags else {}

        self._mlflow_run = None
        self._wandb_run = None

    def __enter__(self) -> "ExperimentLogger":
        if self.use_mlflow:
            mlflow.set_experiment(self.project)
            self._mlflow_run = mlflow.start_run(run_name=self.run_name)
            mlflow.log_params(self.config)
            for k, v in self.tags.items():
                mlflow.set_tag(k, v)

        if self.use_wandb:
            self._wandb_run = wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
                tags=list(self.tags.values()) or None,
                reinit=True,
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self.use_mlflow and self._mlflow_run is not None:
            mlflow.end_run()
        if self.use_wandb and self._wandb_run is not None:
            wandb.finish()

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log a dict of scalar metrics at an optional step."""
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if self.use_mlflow:
            mlflow.log_metrics(clean, step=step)
        if self.use_wandb:
            wandb.log(clean, step=step)

    def log_artifact(self, path: str | Path) -> None:
        """Log a file artifact (model checkpoint, figure, csv...)."""
        path = str(path)
        if self.use_mlflow:
            mlflow.log_artifact(path)
        if self.use_wandb and self._wandb_run is not None:
            wandb.save(path)
