"""Image transforms for PCam.

Histopathology images are rotation- and reflection-symmetric (Veeling et al.,
2018), so standard augmentations include 90° rotations and horizontal/vertical
flips. Color jitter approximates inter-laboratory stain variation.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def _to_tensor_from_hwc(image: np.ndarray) -> torch.Tensor:
    """Convert a uint8 HWC numpy patch to a float32 CHW tensor in [0, 1]."""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor

def identity(x):
    return x

def rotate90(x):
    return torch.rot90(x, 1, dims=(-2, -1))

def rotate180(x):
    return torch.rot90(x, 2, dims=(-2, -1))

def rotate270(x):
    return torch.rot90(x, 3, dims=(-2, -1))
    
def build_train_transform(use_augmentation: bool = True) -> transforms.Compose:
    """Training transform — augmentation can be toggled for ablation."""
    ops: list = [transforms.Lambda(_to_tensor_from_hwc)]
    if use_augmentation:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # 90° rotations are equivariance-respecting for histopathology.
                transforms.RandomChoice(
                    [
                        transforms.Lambda(identity),
                        transforms.Lambda(rotate90),
                        transforms.Lambda(rotate180),
                        transforms.Lambda(rotate270),
                    ]
                ),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
                ),
            ]
        )
    ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)


def build_eval_transform() -> transforms.Compose:
    """Deterministic evaluation transform — no augmentation, only normalisation."""
    return transforms.Compose(
        [
            transforms.Lambda(_to_tensor_from_hwc),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
