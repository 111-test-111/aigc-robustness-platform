"""Tests for chart generation (sample grid, metric bars, radar)."""

from __future__ import annotations

from pathlib import Path

import torch


def test_generate_sample_grid(tmp_path: Path) -> None:
    """Grid of clean vs adversarial images is saved successfully."""
    from src.reporting.charts import generate_sample_grid

    clean = torch.rand(4, 3, 32, 32)
    adv = torch.rand(4, 3, 32, 32)
    path = generate_sample_grid(clean, adv, save_path=tmp_path / "grid.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_generate_sample_grid_with_defended(tmp_path: Path) -> None:
    """Grid including defended samples is saved successfully."""
    from src.reporting.charts import generate_sample_grid

    clean = torch.rand(2, 3, 32, 32)
    adv = torch.rand(2, 3, 32, 32)
    defended = torch.rand(2, 3, 32, 32)
    path = generate_sample_grid(clean, adv, defended, save_path=tmp_path / "grid.png")

    assert path.exists()


def test_generate_sample_grid_max_samples(tmp_path: Path) -> None:
    """Grid respects max_samples limit."""
    from src.reporting.charts import generate_sample_grid

    clean = torch.rand(16, 3, 32, 32)
    adv = torch.rand(16, 3, 32, 32)
    path = generate_sample_grid(clean, adv, save_path=tmp_path / "grid.png", max_samples=3)

    assert path.exists()
    assert path.stat().st_size > 0


def test_generate_metric_bars(tmp_path: Path) -> None:
    """Grouped bar chart is saved successfully."""
    from src.reporting.charts import generate_metric_bars

    m1 = {"ASR": 0.8, "LPIPS": 0.3}
    m2 = {"ASR": 0.5, "LPIPS": 0.1}
    path = generate_metric_bars(
        [m1, m2], ["PGD", "FGSM"], save_path=tmp_path / "bars.png"
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_generate_metric_bars_empty(tmp_path: Path) -> None:
    """Empty metrics list returns path without crashing."""
    from src.reporting.charts import generate_metric_bars

    path = generate_metric_bars([], [], save_path=tmp_path / "bars.png")

    # Empty input returns the path; file may not be written
    assert isinstance(path, Path)


def test_generate_radar(tmp_path: Path) -> None:
    """Radar chart is saved successfully."""
    from src.reporting.charts import generate_radar

    metrics = {"ASR": 0.8, "Robust Acc": 0.6, "LPIPS": 0.3}
    path = generate_radar(metrics, save_path=tmp_path / "radar.png")

    assert path.exists()
    assert path.stat().st_size > 0
