"""Tests for perceptual quality metrics."""

from __future__ import annotations

import inspect

import pytest
import torch

try:
    from src.evaluation.quality_metrics import compute_lpips

    _lpips_available = True
except ImportError:
    _lpips_available = False


lpips_required = pytest.mark.skipif(
    not _lpips_available, reason="lpips package not installed"
)


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------


class TestComputeLpips:
    @lpips_required
    def test_same_image_near_zero(self) -> None:
        """Same image should have LPIPS ~0."""
        torch.manual_seed(0)
        img = torch.rand(2, 3, 64, 64)
        score = compute_lpips(img, img)
        assert score < 0.01

    @lpips_required
    def test_different_images_above_zero(self) -> None:
        """Different images should have LPIPS > 0."""
        torch.manual_seed(0)
        img1 = torch.rand(2, 3, 64, 64)
        torch.manual_seed(1)
        img2 = torch.rand(2, 3, 64, 64)
        score = compute_lpips(img1, img2)
        assert score > 0.01

    @lpips_required
    def test_returns_float(self) -> None:
        """Result should be a plain float."""
        img = torch.rand(1, 3, 32, 32)
        score = compute_lpips(img, img)
        assert isinstance(score, float)

    def test_function_signature(self) -> None:
        """compute_lpips accepts 'clean' and 'adversarial' parameters."""
        sig = inspect.signature(compute_lpips)
        assert "clean" in sig.parameters
        assert "adversarial" in sig.parameters
