"""PatchCamelyon (PCam) dataset loader.

The dataset is distributed as six HDF5 files (train/valid/test × x/y) of 96×96
RGB patches and binary labels. Reference: Veeling et al., "Rotation
Equivariant CNNs for Digital Pathology" (2018). Dataset repo:
https://github.com/basveeling/pcam
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

Split = Literal["train", "valid", "test"]

# Filename convention from the official PCam release.
_X_FILES = {
    "train": "camelyonpatch_level_2_split_train_x.h5",
    "valid": "camelyonpatch_level_2_split_valid_x.h5",
    "test": "camelyonpatch_level_2_split_test_x.h5",
}
_Y_FILES = {
    "train": "camelyonpatch_level_2_split_train_y.h5",
    "valid": "camelyonpatch_level_2_split_valid_y.h5",
    "test": "camelyonpatch_level_2_split_test_y.h5",
}

# ImageNet statistics — appropriate when using ImageNet-pretrained backbones
# such as ResNet50. The custom CNN uses the same normalization for consistency.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PCamDataset(Dataset):
    """In-memory-lazy PatchCamelyon dataset.

    Files are opened with h5py once per worker; indexing reads a single patch
    on demand, which avoids loading 327k patches into RAM.

    Args:
        root: Directory containing the six PCam .h5 files.
        split: One of ``"train"``, ``"valid"``, ``"test"``.
        transform: Optional torchvision-style transform applied to a uint8
            HWC numpy array; should return a torch tensor.
        subset_fraction: If < 1.0, use only the first ``ceil(N * f)`` samples.
            Useful for quick ablation runs and CI smoke tests.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        transform: Callable | None = None,
        subset_fraction: float = 1.0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        x_path = self.root / _X_FILES[split]
        y_path = self.root / _Y_FILES[split]
        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(
                f"PCam files for split '{split}' not found in {self.root}. "
                "Download from https://github.com/basveeling/pcam and place the "
                "six .h5 files in this directory."
            )

        self._x_path = x_path
        self._y_path = y_path
        # File handles are opened lazily per-worker to play nicely with
        # multi-process DataLoaders.
        self._x_file: h5py.File | None = None
        self._y_file: h5py.File | None = None

        with h5py.File(y_path, "r") as f:
            n = f["y"].shape[0]
        self._n = int(np.ceil(n * subset_fraction)) if subset_fraction < 1.0 else n

    def _ensure_open(self) -> None:
        if self._x_file is None:
            self._x_file = h5py.File(self._x_path, "r")
        if self._y_file is None:
            self._y_file = h5py.File(self._y_path, "r")

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        assert self._x_file is not None and self._y_file is not None  # for mypy

        image = self._x_file["x"][idx]  # uint8 HWC 96x96x3
        label = self._y_file["y"][idx]
        # Official labels are shaped (1,1,1) — squeeze to scalar.
        label = int(np.asarray(label).squeeze())

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def class_balance(self) -> dict[int, int]:
        """Return label counts. Useful for sanity-checking class balance."""
        self._ensure_open()
        assert self._y_file is not None
        y = np.asarray(self._y_file["y"]).squeeze()
        return {int(c): int((y == c).sum()) for c in np.unique(y)}
