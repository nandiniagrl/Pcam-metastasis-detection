"""Grad-CAM visualisations for ResNet50 on PCam.

Implements Selvaraju et al.'s class-discriminative localisation by hooking the
last convolutional block of ResNet50 (``layer4``). For each sampled patch we
produce a side-by-side: original image | heatmap | prediction.

Run with ``python -m src.explainability.gradcam_resnet50 --checkpoint ...``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, PCamDataset
from src.data.transforms import build_eval_transform
from src.models.resnet50 import ResNet50Classifier
from src.utils.seed import set_seed


class GradCAM:
    """Minimal Grad-CAM implementation that does not depend on the
    ``grad-cam`` package, to keep the explanation transparent."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output) -> None:  # noqa: ANN001
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:  # noqa: ANN001
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, target_class].sum()
        score.backward()

        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def _denormalise(t: torch.Tensor) -> np.ndarray:
    """Invert ImageNet normalisation for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (t.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM for ResNet50.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--freeze", default="last_block")
    parser.add_argument("--output", default="results/figures/resnet50_gradcam_examples.png")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = PCamDataset(args.data_root, "test", build_eval_transform())
    rng = np.random.default_rng(args.seed)
    # Select half positive + half negative so the figure shows both classes.
    pos_indices = [i for i in rng.integers(0, len(test_ds), size=400) if test_ds[int(i)][1].item() == 1]
    neg_indices = [i for i in rng.integers(0, len(test_ds), size=400) if test_ds[int(i)][1].item() == 0]
    n_each = args.n_samples // 2
    selected = list(pos_indices[:n_each]) + list(neg_indices[:n_each])

    model = ResNet50Classifier(freeze=args.freeze, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    cam = GradCAM(model, model.backbone.layer4[-1])

    fig, axes = plt.subplots(args.n_samples, 3, figsize=(8, 2.6 * args.n_samples))
    for row, idx in enumerate(selected):
        x, y = test_ds[int(idx)]
        x = x.unsqueeze(0).to(device)
        heat = cam(x.clone(), target_class=1)

        logits = model(x)
        prob = float(torch.softmax(logits, dim=1)[0, 1])

        img = _denormalise(x.squeeze(0))
        axes[row, 0].imshow(img); axes[row, 0].set_title(f"input — true={int(y)}"); axes[row, 0].axis("off")
        axes[row, 1].imshow(heat, cmap="jet"); axes[row, 1].set_title("Grad-CAM (class=tumor)"); axes[row, 1].axis("off")
        axes[row, 2].imshow(img); axes[row, 2].imshow(heat, cmap="jet", alpha=0.5)
        axes[row, 2].set_title(f"overlay — P(tumor)={prob:.3f}"); axes[row, 2].axis("off")

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved Grad-CAM examples → {args.output}")


if __name__ == "__main__":
    main()
