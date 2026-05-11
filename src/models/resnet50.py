"""ResNet50 transfer-learning model for PCam.

Uses ImageNet-pretrained weights from torchvision. The final fully-connected
layer is replaced with a 2-class classifier. Freezing strategy is configurable
to support ablation studies (frozen backbone vs. last-block vs. full fine-tune).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models

FreezeStrategy = Literal["none", "all_but_fc", "last_block"]


class ResNet50Classifier(nn.Module):
    """ResNet50 backbone + 2-class head.

    Args:
        num_classes: Output classes (default 2).
        freeze: Which parameters to freeze.
            - ``"none"``: full fine-tuning.
            - ``"all_but_fc"``: freeze backbone, train only the classifier head.
            - ``"last_block"``: freeze everything except ``layer4`` and the head.
        pretrained: Whether to load ImageNet weights.
    """

    def __init__(
        self,
        num_classes: int = 2,
        freeze: FreezeStrategy = "last_block",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        self.freeze_strategy: FreezeStrategy = freeze
        self._apply_freeze(freeze)

    def _apply_freeze(self, freeze: FreezeStrategy) -> None:
        if freeze == "none":
            for p in self.backbone.parameters():
                p.requires_grad = True
            return

        # First freeze everything...
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ...then selectively unfreeze.
        if freeze == "all_but_fc":
            for p in self.backbone.fc.parameters():
                p.requires_grad = True
        elif freeze == "last_block":
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True
            for p in self.backbone.fc.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze strategy: {freeze!r}")

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
