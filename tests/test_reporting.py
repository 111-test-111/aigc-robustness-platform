"""Tests for Markdown report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _make_experiment(tmp_path: Path) -> Path:
    """Create a minimal fake experiment directory with config and metrics."""
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    cfg = OmegaConf.create({
        "task": {"name": "test", "seed": 42, "device": "cpu"},
        "dataset": {
            "name": "cifar10",
            "root": "data",
            "num_samples": 10,
            "image_size": 32,
        },
        "target_model": {"name": "resnet50", "weights": "none"},
        "attacks": [{"name": "fgsm", "eps": 0.03}],
        "defenses": [{"name": "jpeg", "quality": 75}],
        "report": {"output_dir": str(exp_dir), "formats": ["markdown"]},
    })
    OmegaConf.save(cfg, exp_dir / "config.yaml")

    metrics = {
        "fgsm_asr": 0.85,
        "fgsm_vs_jpeg_robust_accuracy": 0.72,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics))

    return exp_dir


def test_generate_report(tmp_path: Path) -> None:
    """Report is generated and contains key experiment information."""
    from src.reporting.markdown_report import generate_report

    exp_dir = _make_experiment(tmp_path)
    report_path = generate_report(exp_dir)

    # File should exist
    assert report_path.exists()
    assert report_path.name == "report.md"

    content = report_path.read_text()

    # Configuration info
    assert "test" in content
    assert "cifar10" in content
    assert "resnet50" in content

    # Attack and defense
    assert "fgsm" in content
    assert "jpeg" in content

    # Metrics (ASR value)
    assert "85.00" in content

    # Conclusion text (ASR > 0.8 -> 显著)
    assert "攻击效果显著" in content

    # Robust accuracy metric
    assert "fgsm_vs_jpeg_robust_accuracy" in content

    # Timestamp
    assert "自动生成于" in content

    # Robustness score section
    assert "综合鲁棒性评分" in content

    # Section headings
    assert "## 1. 实验配置" in content
    assert "## 2. 攻击评估" in content
    assert "## 6. 结论" in content


def test_generate_report_with_figures(tmp_path: Path) -> None:
    """Test report with figures and robustness score."""
    from src.reporting.markdown_report import generate_report

    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    (exp_dir / "figures").mkdir()
    (exp_dir / "figures" / "sample_grid.png").write_bytes(b"fake")

    cfg = OmegaConf.create({
        "task": {"name": "test", "seed": 42, "device": "cpu"},
        "dataset": {
            "name": "cifar10",
            "root": "data",
            "num_samples": 10,
            "image_size": 32,
        },
        "target_model": {"name": "resnet50", "weights": "none"},
        "attacks": [{"name": "fgsm", "eps": 0.03}],
        "defenses": [{"name": "jpeg", "quality": 75}],
        "report": {"output_dir": str(exp_dir), "formats": ["markdown"]},
    })
    OmegaConf.save(cfg, exp_dir / "config.yaml")

    metrics = {"fgsm_asr": 0.85, "fgsm_vs_jpeg_robust_accuracy": 0.72}
    (exp_dir / "metrics.json").write_text(json.dumps(metrics))

    report_path = generate_report(exp_dir)
    content = report_path.read_text()

    # Figure reference present
    assert "sample_grid.png" in content

    # Robustness score present
    assert "综合鲁棒性评分" in content

    # Conclusions present
    assert "结论" in content

    # ASR value in conclusion
    assert "85" in content or "0.85" in content


def test_generate_report_no_figures(tmp_path: Path) -> None:
    """Report omits figure sections when no figures exist."""
    from src.reporting.markdown_report import generate_report

    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    cfg = OmegaConf.create({
        "task": {"name": "test", "seed": 42, "device": "cpu"},
        "dataset": {
            "name": "cifar10",
            "root": "data",
            "num_samples": 10,
            "image_size": 32,
        },
        "target_model": {"name": "resnet50", "weights": "none"},
        "attacks": [{"name": "fgsm", "eps": 0.03}],
        "defenses": [],
        "report": {"output_dir": str(exp_dir), "formats": ["markdown"]},
    })
    OmegaConf.save(cfg, exp_dir / "config.yaml")

    metrics = {"fgsm_asr": 0.30}
    (exp_dir / "metrics.json").write_text(json.dumps(metrics))

    report_path = generate_report(exp_dir)
    content = report_path.read_text()

    # No figure references
    assert "sample_grid.png" not in content
    assert "radar.png" not in content
    assert "metric_bars.png" not in content

    # Robustness score still present
    assert "综合鲁棒性评分" in content


def test_generate_report_low_asr(tmp_path: Path) -> None:
    """Low ASR produces the 有限 conclusion."""
    from src.reporting.markdown_report import generate_report

    exp_dir = tmp_path / "low_asr"
    exp_dir.mkdir()

    cfg = OmegaConf.create({
        "task": {"name": "low", "seed": 0, "device": "cpu"},
        "dataset": {
            "name": "cifar10",
            "root": "data",
            "num_samples": 5,
            "image_size": 32,
        },
        "target_model": {"name": "resnet50", "weights": "none"},
        "attacks": [{"name": "fgsm", "eps": 0.01}],
        "defenses": [],
        "report": {"output_dir": str(exp_dir), "formats": ["markdown"]},
    })
    OmegaConf.save(cfg, exp_dir / "config.yaml")

    metrics = {"fgsm_asr": 0.20}
    (exp_dir / "metrics.json").write_text(json.dumps(metrics))

    report_path = generate_report(exp_dir)
    content = report_path.read_text()

    assert "20.00" in content
    assert "攻击效果有限" in content


def test_generate_report_medium_asr(tmp_path: Path) -> None:
    """Medium ASR (0.5-0.8) produces the 中等 conclusion."""
    from src.reporting.markdown_report import generate_report

    exp_dir = tmp_path / "medium_asr"
    exp_dir.mkdir()

    cfg = OmegaConf.create({
        "task": {"name": "medium", "seed": 0, "device": "cpu"},
        "dataset": {
            "name": "cifar10",
            "root": "data",
            "num_samples": 5,
            "image_size": 32,
        },
        "target_model": {"name": "resnet50", "weights": "none"},
        "attacks": [{"name": "pgd", "eps": 0.02}],
        "defenses": [],
        "report": {"output_dir": str(exp_dir), "formats": ["markdown"]},
    })
    OmegaConf.save(cfg, exp_dir / "config.yaml")

    metrics = {"pgd_asr": 0.65}
    (exp_dir / "metrics.json").write_text(json.dumps(metrics))

    report_path = generate_report(exp_dir)
    content = report_path.read_text()

    assert "65.00" in content
    assert "攻击效果中等" in content


def test_generate_report_markdown_syntax(tmp_path: Path) -> None:
    """Generated report has valid Markdown table structure."""
    from src.reporting.markdown_report import generate_report

    exp_dir = _make_experiment(tmp_path)
    report_path = generate_report(exp_dir)

    content = report_path.read_text()

    # Tables should have proper header separator
    assert "|------|" in content

    # Headings
    assert "# " in content
    assert "## " in content
