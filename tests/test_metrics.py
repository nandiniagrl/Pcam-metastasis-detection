"""Unit tests for the metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import compute_metrics, summarize_runs


def test_perfect_classifier() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_prob)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.roc_auc == 1.0
    assert m.tp == 2 and m.tn == 2 and m.fp == 0 and m.fn == 0


def test_random_classifier() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=1000)
    y_prob = rng.uniform(0, 1, size=1000)
    m = compute_metrics(y_true, y_prob)
    # Random probabilities → AUC near 0.5.
    assert 0.4 < m.roc_auc < 0.6


def test_log_loss_finite_at_extremes() -> None:
    """Predicted probs of exactly 0 or 1 must not yield NaN/inf."""
    y_true = np.array([0, 1])
    y_prob = np.array([0.0, 1.0])
    m = compute_metrics(y_true, y_prob)
    assert np.isfinite(m.nll)


def test_summarize_runs_aggregation() -> None:
    runs = [
        compute_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.6, 0.4])),
        compute_metrics(np.array([0, 1, 1, 0]), np.array([0.2, 0.8, 0.7, 0.3])),
    ]
    s = summarize_runs(runs)
    assert "accuracy" in s
    assert "mean" in s["accuracy"]
    assert s["accuracy"]["mean"] > 0


def test_summarize_runs_empty() -> None:
    assert summarize_runs([]) == {}
