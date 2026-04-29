"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (no network, fast)")
    config.addinivalue_line("markers", "integration: Integration tests (mock backends)")
    config.addinivalue_line("markers", "slow: Slow tests (real models, network)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (full pipeline)")


@pytest.fixture
def device() -> torch.device:
    """Return the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Return CPU device."""
    return torch.device("cpu")


@pytest.fixture
def small_image_batch() -> torch.Tensor:
    """Return a small batch of random images for testing."""
    return torch.rand(2, 3, 32, 32)


@pytest.fixture
def small_labels() -> torch.Tensor:
    """Return labels for a small batch."""
    return torch.randint(0, 10, (2,))


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Return a temporary output directory for test artifacts."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir
