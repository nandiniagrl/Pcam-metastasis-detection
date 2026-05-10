"""A small custom CNN baseline for PCam.

This is a deliberately simple architecture — five convolutional blocks with
batch normalization and pooling — intended as a deep-learning reference point
without transfer learning. It serves as the bridge between the classical
logistic-regression baseline and the ResNet50 transfer-learning model.
"""

from __future__ import annotations

import torch
from torch import nn


class CustomCNN(nn.Module):
    """Compact CNN for 96×96 RGB patches → 2-class logits."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),   # 96 → 48
            block(32, 64),  # 48 → 24
            block(64, 128), # 24 → 12
            block(128, 256),# 12 → 6
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
