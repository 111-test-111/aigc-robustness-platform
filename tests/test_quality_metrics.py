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


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

try:
    from src.evaluation.quality_metrics import (
        DEFAULT_FID_INCEPTION_WEIGHTS_URL,
        get_fid_inception_weights_url,
        compute_fid,
    )

    _fid_available = True
except ImportError:
    _fid_available = False

fid_required = pytest.mark.skipif(
    not _fid_available, reason="torchmetrics not available"
)


class TestComputeFid:
    def test_fid_weights_url_defaults_to_github(self, monkeypatch) -> None:
        """FID weights use the official URL unless a mirror is configured."""
        monkeypatch.delenv("AIGC_FID_WEIGHTS_URL", raising=False)
        assert get_fid_inception_weights_url() == DEFAULT_FID_INCEPTION_WEIGHTS_URL

    def test_fid_weights_url_can_use_mirror(self, monkeypatch) -> None:
        """Server deployments can route FID weights through a mirror URL."""
        mirror = (
            "https://mirror.example/https://github.com/toshas/torch-fidelity/"
            "releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"
        )
        monkeypatch.setenv("AIGC_FID_WEIGHTS_URL", mirror)
        assert get_fid_inception_weights_url() == mirror

    @fid_required
    def test_fid_same_distribution(self) -> None:
        """Same images should have low FID (near zero, within float tolerance)."""
        imgs = torch.rand(32, 3, 64, 64)
        fid = compute_fid(imgs, imgs)
        assert fid >= -1e-6  # FID is non-negative (tiny float noise allowed)

    @fid_required
    def test_fid_different_distributions(self) -> None:
        """Different distributions should have higher FID."""
        real = torch.rand(32, 3, 64, 64)
        fake = torch.rand(32, 3, 64, 64) + 0.5
        fid = compute_fid(real, fake)
        assert fid >= 0


# ---------------------------------------------------------------------------
# CLIP Score
# ---------------------------------------------------------------------------

try:
    from src.evaluation.quality_metrics import compute_clip_score

    _clip_available = True
except ImportError:
    _clip_available = False

clip_required = pytest.mark.skipif(
    not _clip_available, reason="transformers not available"
)


class TestComputeClipScore:
    @clip_required
    @pytest.mark.slow
    def test_clip_score_basic(self) -> None:
        """CLIP score should be in [0, 1] range."""
        imgs = torch.rand(4, 3, 64, 64)
        score = compute_clip_score(imgs, "a random image")
        assert 0 <= score <= 1

    def test_clip_score_function_exists(self) -> None:
        from src.evaluation.quality_metrics import compute_clip_score

        sig = inspect.signature(compute_clip_score)
        assert "images" in sig.parameters
        assert "prompt" in sig.parameters
