"""Tests for model_zoo registry and classifier loading."""

import pytest
import torch

from src.model_zoo.registry import MODEL_REGISTRY
from src.model_zoo.classifiers import load_classifier


def test_registry_has_resnet50():
    """Verify 'resnet50' is registered after importing classifiers."""
    assert "resnet50" in MODEL_REGISTRY


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
