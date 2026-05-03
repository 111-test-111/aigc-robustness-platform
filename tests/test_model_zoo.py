"""Tests for model_zoo registry and classifier loading."""

import pytest
import torch

from src.model_zoo.registry import MODEL_REGISTRY
from src.model_zoo.classifiers import (
    DEFAULT_TORCHVISION_WEIGHTS_BASE_URL,
    get_torchvision_weights_base_url,
    get_torchvision_weights_url,
    _remap_legacy_densenet_state_dict,
    load_torchvision_state_dict,
    load_classifier,
)


def test_registry_has_resnet50():
    """Verify 'resnet50' is registered after importing classifiers."""
    assert "resnet50" in MODEL_REGISTRY


def test_torchvision_weights_url_defaults_to_pytorch(monkeypatch):
    """Torchvision classifier weights default to PyTorch's official host."""
    monkeypatch.delenv("AIGC_TORCHVISION_WEIGHTS_BASE_URL", raising=False)
    assert get_torchvision_weights_base_url() == DEFAULT_TORCHVISION_WEIGHTS_BASE_URL
    assert get_torchvision_weights_url("densenet121-a639ec97.pth") == (
        "https://download.pytorch.org/models/densenet121-a639ec97.pth"
    )


def test_torchvision_weights_url_can_use_mirror(monkeypatch):
    """Server deployments can route torchvision weights through a mirror base."""
    monkeypatch.setenv(
        "AIGC_TORCHVISION_WEIGHTS_BASE_URL", "https://mirror.example/pytorch/models/"
    )
    assert get_torchvision_weights_url("densenet121-a639ec97.pth") == (
        "https://mirror.example/pytorch/models/densenet121-a639ec97.pth"
    )


def test_load_torchvision_state_dict_uses_mirror(monkeypatch):
    """State-dict downloads should use the configured mirror and cache filename."""
    calls = {}

    def fake_load_state_dict_from_url(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return {}

    monkeypatch.setenv(
        "AIGC_TORCHVISION_WEIGHTS_BASE_URL", "https://mirror.example/pytorch/models"
    )
    monkeypatch.setattr(
        torch.hub, "load_state_dict_from_url", fake_load_state_dict_from_url
    )

    assert load_torchvision_state_dict("alexnet-owt-7be5be79.pth") == {}
    assert calls["url"] == (
        "https://mirror.example/pytorch/models/alexnet-owt-7be5be79.pth"
    )
    assert calls["kwargs"]["file_name"] == "alexnet-owt-7be5be79.pth"
    assert calls["kwargs"]["check_hash"]


def test_legacy_densenet_state_dict_keys_are_remapped():
    """DenseNet checkpoints with old dotted keys should load on new modules."""
    state_dict = {
        "features.denseblock1.denselayer1.norm.1.weight": torch.ones(1),
        "features.denseblock1.denselayer1.norm.1.bias": torch.zeros(1),
        "features.denseblock1.denselayer1.conv.2.weight": torch.ones(1, 1, 1, 1),
        "classifier.weight": torch.ones(1, 1),
    }

    remapped = _remap_legacy_densenet_state_dict(state_dict)

    assert "features.denseblock1.denselayer1.norm1.weight" in remapped
    assert "features.denseblock1.denselayer1.norm1.bias" in remapped
    assert "features.denseblock1.denselayer1.conv2.weight" in remapped
    assert "features.denseblock1.denselayer1.norm.1.weight" not in remapped
    assert "classifier.weight" in remapped


def test_load_classifier_resnet50():
    """Load ResNet-50 and verify it is an nn.Module in eval mode."""
    model = load_classifier("resnet50", weights="none")
    assert isinstance(model, torch.nn.Module)
    assert not model.training


def test_load_classifier_output_shape():
    """Verify ResNet-50 produces correct logits shape for a (1, 3, 224, 224) input."""
    model = load_classifier("resnet50", weights="none")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 1000)


def test_load_classifier_unknown_raises():
    """Verify ValueError is raised for an unregistered model name."""
    with pytest.raises(ValueError, match="Unknown classifier: nonexistent"):
        load_classifier("nonexistent")


def test_load_classifier_device():
    """Verify model is placed on the specified device."""
    device = torch.device("cpu")
    model = load_classifier("resnet50", weights="none", device=device)
    param = next(model.parameters())
    assert param.device == device


def test_load_vit_b_16_output_shape():
    """Verify ViT-B/16 produces correct logits shape for a (1, 3, 224, 224) input."""
    model = load_classifier("vit_b_16", weights="none")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 1000)


def test_load_densenet121_output_shape():
    """Verify DenseNet-121 produces correct logits shape for a (1, 3, 224, 224) input."""
    model = load_classifier("densenet121", weights="none")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 1000)


def test_load_vit_b_16_eval_mode():
    """Verify ViT-B/16 is in eval mode after loading."""
    model = load_classifier("vit_b_16", weights="none")
    assert not model.training


def test_load_densenet121_eval_mode():
    """Verify DenseNet-121 is in eval mode after loading."""
    model = load_classifier("densenet121", weights="none")
    assert not model.training


def test_invalid_model_name_raises():
    """Verify ValueError is raised for an unregistered model name."""
    with pytest.raises(ValueError, match="Unknown classifier: nonexistent"):
        load_classifier("nonexistent")
