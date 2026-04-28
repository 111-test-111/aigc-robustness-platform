"""Tests for Gaussian blur defense."""

import pytest
import torch

from src.defense_engine.base import DefenseResult
from src.defense_engine.blur import GaussianBlurDefense


@pytest.fixture
def defense() -> GaussianBlurDefense:
    return GaussianBlurDefense()


@pytest.fixture
def sample_batch() -> torch.Tensor:
    """Random (2, 3, 224, 224) batch in [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(2, 3, 224, 224)


class TestBlurOutputShape:
    def test_preserves_batch_shape(
        self, defense: GaussianBlurDefense, sample_batch: torch.Tensor
    ) -> None:
        result = defense.apply(sample_batch, config={})
        assert result.defended.shape == sample_batch.shape

    def test_single_image_shape(self, defense: GaussianBlurDefense) -> None:
        batch = torch.rand(1, 3, 64, 64)
        result = defense.apply(batch, config={})
        assert result.defended.shape == (1, 3, 64, 64)


class TestBlurOutputRange:
    def test_output_in_unit_range(
        self, defense: GaussianBlurDefense, sample_batch: torch.Tensor
    ) -> None:
        result = defense.apply(sample_batch, config={})
        assert result.defended.min() >= 0.0
        assert result.defended.max() <= 1.0


class TestBlurChangesImage:
    def test_defended_differs_from_input(
        self, defense: GaussianBlurDefense, sample_batch: torch.Tensor
    ) -> None:
        result = defense.apply(sample_batch, config={})
        l2_dist = torch.norm(result.defended - sample_batch)
        assert l2_dist > 0.0, "Gaussian blur should alter the image"


class TestBlurKernelEffect:
    def test_larger_kernel_more_blur(
        self, defense: GaussianBlurDefense, sample_batch: torch.Tensor
    ) -> None:
        small = defense.apply(sample_batch, config={"kernel_size": 3, "sigma": 0.5})
        large = defense.apply(sample_batch, config={"kernel_size": 11, "sigma": 2.0})
        l2_small = torch.norm(small.defended - sample_batch)
        l2_large = torch.norm(large.defended - sample_batch)
        assert l2_large > l2_small, (
            f"Larger kernel should cause more blur (larger L2), "
            f"got large={l2_large:.4f} <= small={l2_small:.4f}"
        )


class TestBlurLatency:
    def test_latency_recorded(
        self, defense: GaussianBlurDefense, sample_batch: torch.Tensor
    ) -> None:
        result = defense.apply(sample_batch, config={})
        assert isinstance(result, DefenseResult)
        assert result.latency_sec > 0.0
