"""Tests for the diffusion purification defense."""

import torch

from src.defense_engine.base import Defense
from src.defense_engine.diffusion_purification import DiffusionPurificationDefense


def test_class_exists():
    """DiffusionPurificationDefense is importable and exposes expected name."""
    assert DiffusionPurificationDefense.name == "diffusion_purification"


def test_inherits_from_defense():
    """Class must implement the Defense ABC."""
    assert issubclass(DiffusionPurificationDefense, Defense)


def test_output_shape_preserved():
    """Defended batch must have the same shape as the input batch."""
    defense = DiffusionPurificationDefense()
    batch = torch.rand(2, 3, 32, 32)
    result = defense.apply(batch, {"backend": "mock", "noise_level": 0.1, "steps": 3})
    assert result.defended.shape == batch.shape


def test_output_range_clamped():
    """Pixel values must remain within [0, 1]."""
    defense = DiffusionPurificationDefense()
    batch = torch.rand(2, 3, 32, 32)
    result = defense.apply(batch, {"backend": "mock", "noise_level": 0.1, "steps": 3})
    assert result.defended.min() >= 0
    assert result.defended.max() <= 1


def test_latency_recorded():
    """Latency must be a positive float."""
    defense = DiffusionPurificationDefense()
    batch = torch.rand(2, 3, 32, 32)
    result = defense.apply(batch, {"backend": "mock", "noise_level": 0.1, "steps": 3})
    assert result.latency_sec > 0


def test_image_is_modified():
    """Defence must change the image (noise + denoise path)."""
    defense = DiffusionPurificationDefense()
    batch = torch.rand(2, 3, 32, 32)
    result = defense.apply(batch, {"backend": "mock", "noise_level": 0.2, "steps": 3})
    diff = (result.defended - batch).abs().mean().item()
    assert diff > 0


def test_different_noise_levels_affect_output():
    """Higher noise should produce a different result than lower noise."""
    defense = DiffusionPurificationDefense()
    batch = torch.rand(1, 3, 32, 32)
    result_low = defense.apply(batch, {"backend": "mock", "noise_level": 0.01, "steps": 3})
    result_high = defense.apply(batch, {"backend": "mock", "noise_level": 0.1, "steps": 3})
    assert not torch.equal(result_low.defended, result_high.defended)


def test_single_image_batch():
    """Should work with a batch size of 1."""
    defense = DiffusionPurificationDefense()
    batch = torch.rand(1, 3, 32, 32)
    result = defense.apply(batch, {"backend": "mock", "noise_level": 0.1, "steps": 3})
    assert result.defended.shape == (1, 3, 32, 32)
    assert result.latency_sec > 0


def test_config_defaults_used():
    """Omitting config keys should still work with defaults."""
    defense = DiffusionPurificationDefense()
    batch = torch.rand(1, 3, 16, 16)
    result = defense.apply(batch, {"backend": "mock"})
    assert result.defended.shape == batch.shape
    assert result.latency_sec > 0
