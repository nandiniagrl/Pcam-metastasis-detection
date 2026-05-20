"""Smoke tests for model architectures."""

from __future__ import annotations

import torch

from src.models.custom_cnn import CustomCNN
from src.models.resnet50 import ResNet50Classifier


def test_custom_cnn_forward_shape() -> None:
    model = CustomCNN(num_classes=2)
    x = torch.randn(4, 3, 96, 96)
    out = model(x)
    assert out.shape == (4, 2)


def test_resnet50_forward_shape() -> None:
    model = ResNet50Classifier(pretrained=False, freeze="all_but_fc")
    x = torch.randn(2, 3, 96, 96)
    out = model(x)
    assert out.shape == (2, 2)


def test_resnet50_freeze_modes() -> None:
    """Verify each freeze strategy exposes a sensible number of trainable params."""
    full = ResNet50Classifier(pretrained=False, freeze="none")
    head_only = ResNet50Classifier(pretrained=False, freeze="all_but_fc")
    last_block = ResNet50Classifier(pretrained=False, freeze="last_block")

    p_full = full.trainable_parameter_count()
    p_head = head_only.trainable_parameter_count()
    p_last = last_block.trainable_parameter_count()

    # Strict ordering: head < last_block < full.
    assert p_head < p_last < p_full
