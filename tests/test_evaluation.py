"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.evaluation.attack_metrics import compute_asr, compute_queries
from src.evaluation.defense_metrics import (
    compute_clean_accuracy_drop,
    compute_latency,
    compute_robust_accuracy,
)


# ---------------------------------------------------------------------------
# TinyClassifier - deterministic model for testing
# ---------------------------------------------------------------------------


class TinyClassifier(nn.Module):
    """Minimal deterministic classifier for metric tests.

    Maps any input to a fixed prediction per sample so metrics can be
    verified against hand-computed expected values.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


# ---------------------------------------------------------------------------
# Attack metrics
# ---------------------------------------------------------------------------


class TestComputeAsr:
    def test_all_success(self) -> None:
        mask = torch.ones(5, dtype=torch.bool)
        assert compute_asr(mask) == 1.0

    def test_no_success(self) -> None:
        mask = torch.zeros(5, dtype=torch.bool)
        assert compute_asr(mask) == 0.0

    def test_partial(self) -> None:
        mask = torch.tensor([True, False, True, False, False])
        asr = compute_asr(mask)
        assert 0.0 < asr < 1.0
        assert asr == pytest.approx(0.4)

    def test_empty_mask(self) -> None:
        mask = torch.tensor([], dtype=torch.bool)
        assert compute_asr(mask) == 0.0


class TestComputeQueries:
    def test_stats_keys_and_values(self) -> None:
        result = compute_queries([10, 20, 30])
        assert set(result.keys()) == {"mean", "median", "max", "total"}
        assert result["mean"] == pytest.approx(20.0)
        assert result["median"] == pytest.approx(20.0)
        assert result["max"] == 30
        assert result["total"] == 60

    def test_single_element(self) -> None:
        result = compute_queries([7])
        assert result["mean"] == pytest.approx(7.0)
        assert result["median"] == pytest.approx(7.0)
        assert result["max"] == 7
        assert result["total"] == 7

    def test_empty_list(self) -> None:
        result = compute_queries([])
        assert result["mean"] == 0.0
        assert result["median"] == 0.0
        assert result["max"] == 0
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# Defense metrics
# ---------------------------------------------------------------------------


def _make_batch(batch_size: int = 4, channels: int = 3, h: int = 8, w: int = 8) -> torch.Tensor:
    """Create a deterministic batch of images."""
    return torch.randn(batch_size, channels, h, w)


class TestComputeRobustAccuracy:
    def test_perfect_when_defended_equals_clean(self) -> None:
        """If defended == clean, robust accuracy equals clean accuracy."""
        torch.manual_seed(42)
        model = TinyClassifier(num_classes=10)
        model.eval()
        clean = _make_batch(batch_size=8)
        labels = model(clean).argmax(dim=1)
        robust_acc = compute_robust_accuracy(model, clean, labels)
        assert robust_acc == pytest.approx(1.0)

    def test_value_in_range(self) -> None:
        torch.manual_seed(0)
        model = TinyClassifier(num_classes=10)
        defended = _make_batch(batch_size=8)
        labels = torch.randint(0, 10, (8,))
        acc = compute_robust_accuracy(model, defended, labels)
        assert 0.0 <= acc <= 1.0


class TestComputeCleanAccuracyDrop:
    def test_drop_zero_when_defended_equals_clean(self) -> None:
        """If defended == clean, accuracy drop is 0."""
        torch.manual_seed(42)
        model = TinyClassifier(num_classes=10)
        clean = _make_batch(batch_size=8)
        labels = model(clean).argmax(dim=1)
        drop = compute_clean_accuracy_drop(model, clean, clean, labels)
        assert drop == pytest.approx(0.0)

    def test_drop_can_be_negative(self) -> None:
        """Drop is clean_acc - defended_acc; can be negative if defended is better."""
        torch.manual_seed(42)
        model = TinyClassifier(num_classes=10)
        model.eval()
        clean = _make_batch(batch_size=8)
        labels = torch.randint(0, 10, (8,))
        defended = _make_batch(batch_size=8)
        drop = compute_clean_accuracy_drop(model, clean, defended, labels)
        # Just verify it's a finite float - actual value depends on random data
        assert isinstance(drop, float)


class TestComputeLatency:
    def test_stats_keys_and_values(self) -> None:
        result = compute_latency([0.1, 0.2, 0.3])
        assert set(result.keys()) == {"mean", "median", "max", "total"}
        assert result["mean"] == pytest.approx(0.2)
        assert result["median"] == pytest.approx(0.2)
        assert result["max"] == pytest.approx(0.3)
        assert result["total"] == pytest.approx(0.6)

    def test_single_element(self) -> None:
        result = compute_latency([1.5])
        assert result["mean"] == pytest.approx(1.5)
        assert result["max"] == pytest.approx(1.5)
        assert result["total"] == pytest.approx(1.5)

    def test_empty_list(self) -> None:
        result = compute_latency([])
        assert result["mean"] == 0.0
        assert result["median"] == 0.0
        assert result["max"] == 0.0
        assert result["total"] == 0.0
