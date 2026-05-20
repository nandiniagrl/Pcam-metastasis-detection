"""Verify that set_seed produces reproducible randomness."""

from __future__ import annotations

import numpy as np
import torch

from src.utils.seed import set_seed


def test_numpy_reproducibility() -> None:
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_torch_reproducibility() -> None:
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    torch.testing.assert_close(a, b)
