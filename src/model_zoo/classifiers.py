"""Classifier model loaders."""

import torch
import torch.nn as nn
from torchvision import models

from src.model_zoo.registry import MODEL_REGISTRY, register


@register("resnet50")
def load_resnet50(
    weights: str = "imagenet",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    if weights == "imagenet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50(weights=None)
    return model.eval().to(device)


def load_classifier(
    name: str,
    weights: str = "imagenet",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a classifier by name from the registry.

    Args:
        name: Model name (e.g. "resnet50").
        weights: Weight preset (e.g. "imagenet" or "none").
        device: Target device.

    Returns:
        An nn.Module in eval mode on the specified device.

    Raises:
        ValueError: If the model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown classifier: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](weights=weights, device=device)
