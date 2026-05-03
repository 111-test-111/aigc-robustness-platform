"""Classifier model loaders."""

import os
from urllib.parse import urlparse

import torch
import torch.nn as nn
from torchvision import models

from src.model_zoo.registry import MODEL_REGISTRY, register
from src.progress import third_party_progress_enabled

TORCHVISION_WEIGHTS_BASE_URL_ENV = "AIGC_TORCHVISION_WEIGHTS_BASE_URL"
DEFAULT_TORCHVISION_WEIGHTS_BASE_URL = "https://download.pytorch.org/models"


class ImageNetNormalizeWrapper(nn.Module):
    """Wrap an ImageNet classifier while keeping external inputs in [0, 1]."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def get_torchvision_weights_base_url() -> str:
    """Return the base URL used for torchvision classifier weights."""
    return os.environ.get(TORCHVISION_WEIGHTS_BASE_URL_ENV, "").strip() or (
        DEFAULT_TORCHVISION_WEIGHTS_BASE_URL
    )


def get_torchvision_weights_url(filename: str) -> str:
    """Build a torchvision weight URL from a configurable mirror base."""
    return f"{get_torchvision_weights_base_url().rstrip('/')}/{filename}"


def _weights_filename(weights_enum) -> str:
    """Extract the canonical checkpoint filename from a torchvision weight enum."""
    return os.path.basename(urlparse(weights_enum.url).path)


def _load_state_dict_from_weights(weights_enum) -> dict:
    """Load a torchvision state dict, honoring the configured mirror base."""
    return load_torchvision_state_dict(_weights_filename(weights_enum))


def load_torchvision_state_dict(filename: str) -> dict:
    """Load a torchvision checkpoint by filename from the configured base URL."""
    return torch.hub.load_state_dict_from_url(
        get_torchvision_weights_url(filename),
        map_location="cpu",
        progress=third_party_progress_enabled(),
        check_hash=True,
        file_name=filename,
    )


@register("resnet50")
def load_resnet50(
    weights: str = "imagenet",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    if weights == "imagenet":
        weights_enum = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=None)
        model.load_state_dict(_load_state_dict_from_weights(weights_enum))
        model = ImageNetNormalizeWrapper(model)
    else:
        model = models.resnet50(weights=None)
    return model.eval().to(device)


@register("vit_b_16")
def load_vit_b_16(
    weights: str = "imagenet",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    if weights == "imagenet":
        weights_enum = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=None)
        model.load_state_dict(_load_state_dict_from_weights(weights_enum))
        model = ImageNetNormalizeWrapper(model)
    else:
        model = models.vit_b_16(weights=None)
    return model.eval().to(device)


@register("densenet121")
def load_densenet121(
    weights: str = "imagenet",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    if weights == "imagenet":
        weights_enum = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=None)
        model.load_state_dict(_load_state_dict_from_weights(weights_enum))
        model = ImageNetNormalizeWrapper(model)
    else:
        model = models.densenet121(weights=None)
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
