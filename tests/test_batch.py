"""Tests for batch experiment runner and comparison report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


@pytest.mark.integration
def test_run_batch(tmp_path: Path) -> None:
    """Test batch experiment runner with two configs."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    for i, name in enumerate(["exp_a", "exp_b"]):
        cfg = OmegaConf.create({
            "task": {"name": name, "seed": i, "device": "cpu"},
            "dataset": {
                "name": "cifar10",
                "root": str(tmp_path / "data"),
                "num_samples": 2,
                "image_size": 32,
            },
            "target_model": {
                "type": "classifier",
                "name": "resnet50",
                "weights": "none",
            },
            "attacks": [{"name": "fgsm", "eps": 0.03 + i * 0.02}],
            "defenses": [{"name": "jpeg", "quality": 75}],
            "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
            "report": {
                "output_dir": str(tmp_path / "reports" / name),
                "formats": ["markdown"],
            },
        })
        OmegaConf.save(cfg, config_dir / f"{name}.yaml")

    result = runner.invoke(app, ["run-batch", str(config_dir)])

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "exp_a" / "metrics.json").exists()
    assert (tmp_path / "reports" / "exp_b" / "metrics.json").exists()
    assert (tmp_path / "reports" / "comparison" / "comparison.json").exists()


@pytest.mark.integration
def test_run_batch_comparison_charts(tmp_path: Path) -> None:
    """Test that comparison bar chart and radar are generated."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    for i, name in enumerate(["exp_1", "exp_2"]):
        cfg = OmegaConf.create({
            "task": {"name": name, "seed": i, "device": "cpu"},
            "dataset": {
                "name": "cifar10",
                "root": str(tmp_path / "data"),
                "num_samples": 2,
                "image_size": 32,
            },
            "target_model": {
                "type": "classifier",
                "name": "resnet50",
                "weights": "none",
            },
            "attacks": [{"name": "fgsm", "eps": 0.05}],
            "defenses": [{"name": "jpeg", "quality": 50}],
            "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
            "report": {
                "output_dir": str(tmp_path / "reports" / name),
                "formats": ["markdown"],
            },
        })
        OmegaConf.save(cfg, config_dir / f"{name}.yaml")

    result = runner.invoke(app, ["run-batch", str(config_dir)])

    assert result.exit_code == 0

    comp_dir = tmp_path / "reports" / "comparison"
    assert (comp_dir / "figures" / "comparison_bars.png").exists()
    assert (comp_dir / "figures" / "radar_exp_1.png").exists()


def test_run_batch_empty_directory(tmp_path: Path) -> None:
    """Batch runner exits with error when no configs found."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = runner.invoke(app, ["run-batch", str(empty_dir)])

    assert result.exit_code == 1
    assert "未找到配置文件" in result.output


def test_run_batch_nonexistent_directory(tmp_path: Path) -> None:
    """Batch runner exits with error when directory does not exist."""
    fake_dir = tmp_path / "nonexistent"

    result = runner.invoke(app, ["run-batch", str(fake_dir)])

    assert result.exit_code != 0


def test_run_batch_yml_extension(tmp_path: Path) -> None:
    """Batch runner picks up .yml files in addition to .yaml."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    cfg = OmegaConf.create({
        "task": {"name": "yml_test", "seed": 0, "device": "cpu"},
        "dataset": {
            "name": "cifar10",
            "root": str(tmp_path / "data"),
            "num_samples": 2,
            "image_size": 32,
        },
        "target_model": {
            "type": "classifier",
            "name": "resnet50",
            "weights": "none",
        },
        "attacks": [{"name": "fgsm", "eps": 0.03}],
        "defenses": [{"name": "jpeg", "quality": 75}],
        "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
        "report": {
            "output_dir": str(tmp_path / "reports" / "yml_test"),
            "formats": ["markdown"],
        },
    })
    OmegaConf.save(cfg, config_dir / "config.yml")

    result = runner.invoke(app, ["run-batch", str(config_dir)])

    assert result.exit_code == 0
    assert (tmp_path / "reports" / "yml_test" / "metrics.json").exists()


@pytest.mark.integration
def test_run_batch_comparison_json_content(tmp_path: Path) -> None:
    """Test that comparison.json contains metrics keyed by experiment name."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    for i, name in enumerate(["alpha", "beta"]):
        cfg = OmegaConf.create({
            "task": {"name": name, "seed": i, "device": "cpu"},
            "dataset": {
                "name": "cifar10",
                "root": str(tmp_path / "data"),
                "num_samples": 2,
                "image_size": 32,
            },
            "target_model": {
                "type": "classifier",
                "name": "resnet50",
                "weights": "none",
            },
            "attacks": [{"name": "fgsm", "eps": 0.05}],
            "defenses": [{"name": "jpeg", "quality": 75}],
            "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
            "report": {
                "output_dir": str(tmp_path / "reports" / name),
                "formats": ["markdown"],
            },
        })
        OmegaConf.save(cfg, config_dir / f"{name}.yaml")

    result = runner.invoke(app, ["run-batch", str(config_dir)])
    assert result.exit_code == 0

    comp_json = tmp_path / "reports" / "comparison" / "comparison.json"
    assert comp_json.exists()

    combined = json.loads(comp_json.read_text())
    assert "alpha" in combined
    assert "beta" in combined
    assert isinstance(combined["alpha"], dict)
    assert isinstance(combined["beta"], dict)
