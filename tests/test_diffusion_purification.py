"""Tests for the diffusion purification defense."""

import torch
from PIL import Image

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


class FakeImg2ImgResult:
    def __init__(self, images):
        self.images = images


class FakeImg2ImgPipe:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, *, prompt, image, **kwargs):
        self.batch_sizes.append(len(image))
        assert len(prompt) == len(image)
        return FakeImg2ImgResult([
            Image.new("RGB", img.size, color=(idx, idx, idx))
            for idx, img in enumerate(image)
        ])


def test_sd_purification_uses_configured_batches(monkeypatch):
    """SD purification should denoise chunks, not call the pipeline per image."""
    import src.model_zoo.generators as generators

    pipe = FakeImg2ImgPipe()
    monkeypatch.setattr(generators, "load_sd_pipeline", lambda *args, **kwargs: pipe)

    defense = DiffusionPurificationDefense()
    batch = torch.rand(5, 3, 16, 16)
    result = defense._denoise_with_sd(
        batch,
        model_id="fake",
        steps=10,
        device=torch.device("cpu"),
        sd_batch_size=2,
    )

    assert pipe.batch_sizes == [2, 2, 1]
    assert result.shape == batch.shape
