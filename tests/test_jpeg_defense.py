"""Tests for JPEG compression defense."""

import pytest
import torch

from src.defense_engine.base import DefenseResult
from src.defense_engine.jpeg import JPEGDefense


@pytest.fixture
def defense() -> JPEGDefense:
    return JPEGDefense()


@pytest.fixture
def sample_batch() -> torch.Tensor:
    """Random (2, 3, 224, 224) batch in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(2, 3, 224, 224)


class TestJPEGOutputShape:
    def test_preserves_batch_shape(
        self, defense: JPEGDefense, sample_batch: torch.Tensor
    ) -> None:
        result = defense.apply(sample_batch, config={})
        assert result.defended.shape == sample_batch.shape

    def test_single_image_shape(self, defense: JPEGDefense) -> None:
        batch = torch.rand(1, 3, 64, 64)
        result = defense.apply(batch, config={})
        assert result.defended.shape == (1, 3, 64, 64)


class TestJPEGOutputRange:
    def test_output_in_unit_range(
        self, defense: JPEGDefense, sample_batch: torch.Tensor
    ) -> None:
        result = defense.apply(sample_batch, config={})
        assert result.defended.min() >= 0.0
        assert result.defended.max() <= 1.0


class TestJPEGChangesImage:
    def test_defended_differs_from_input(
        self, defense: JPEGDefense, sample_batch: torch.Tensor
    ) -> None:
        result = defense.apply(sample_batch, config={})
        l2_dist = torch.norm(result.defended - sample_batch)
        assert l2_dist > 0.0, "JPEG defense should alter the image"


class TestJPEGQualityEffect:
    def test_lower_quality带来更多扰动(
        self, defense: JPEGDefense, sample_batch: torch.Tensor
    ) -> None:
        high = defense.apply(sample_batch, config={"quality": 95})
        low = defense.apply(sample_batch, config={"quality": 10})
        l2_high = torch.norm(high.defended - sample_batch)
        l2_low = torch.norm(low.defended - sample_batch)
        assert l2_low > l2_high, (
            f"Lower quality should cause larger distortion, "
            f"got low={l2_low:.4f} <= high={l2_high:.4f}"
        )


class TestJPEGLatency:
    def test_latency_recorded(
        self, defense: JPEGDefense, sample_batch: torch.Tensor
    ) -> None:
        result = defense.apply(sample_batch, config={})
        assert isinstance(result, DefenseResult)
        assert result.latency_sec > 0.0
