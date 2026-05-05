"""YAML-based configuration loading using OmegaConf for typed access."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file as an OmegaConf DictConfig.

    Args:
        path: Path to a YAML config file.

    Returns:
        Parsed configuration object supporting dot-access (``cfg.training.lr``).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = OmegaConf.load(path)
    return cfg  # type: ignore[return-value]


def merge_overrides(cfg: DictConfig, overrides: list[str] | None) -> DictConfig:
    """Merge command-line dotlist overrides into a config (e.g. ``training.lr=1e-4``)."""
    if not overrides:
        return cfg
    override_cfg = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(cfg, override_cfg)  # type: ignore[return-value]
