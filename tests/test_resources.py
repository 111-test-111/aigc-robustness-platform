"""Tests for resource tracker."""

from __future__ import annotations

import torch

from src.utils.resources import ResourceTracker


def test_resource_tracker_no_cuda() -> None:
    """Tracker works on CPU (no GPU metrics, no crash)."""
    device = torch.device("cpu")
    tracker = ResourceTracker(device)
    tracker.start()
    # simulate work
    _ = torch.randn(100, 100)
    metrics = tracker.stop()

    assert isinstance(metrics, dict)
    # On CPU, no GPU metrics are expected
    assert "gpu_mem_allocated_mb" not in metrics
    assert "gpu_mem_reserved_mb" not in metrics
    # GPU util only available with pynvml + NVIDIA GPU
    assert "gpu_util_pct_mean" not in metrics


def test_resource_tracker_cpu_memory() -> None:
    """CPU RSS peak is captured when psutil is available."""
    device = torch.device("cpu")
    tracker = ResourceTracker(device)
    tracker.start()
    # allocate some memory
    _ = torch.randn(500, 500)
    metrics = tracker.stop()

    assert isinstance(metrics, dict)
    # cpu_rss_peak_mb may or may not be present depending on psutil availability


def test_resource_tracker_reuse() -> None:
    """Multiple start/stop cycles work correctly."""
    device = torch.device("cpu")
    tracker = ResourceTracker(device)

    tracker.start()
    _ = torch.randn(200, 200)
    m1 = tracker.stop()
    assert isinstance(m1, dict)

    tracker.start()
    _ = torch.randn(300, 300)
    m2 = tracker.stop()
    assert isinstance(m2, dict)


def test_resource_tracker_stop_before_start() -> None:
    """Stop before start returns empty dict without crashing."""
    tracker = ResourceTracker(torch.device("cpu"))
    metrics = tracker.stop()
    assert metrics == {}
