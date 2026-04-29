"""End-to-end test for the task runner pipeline."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image

from src.task_runner import _load_dataset, run_experiment


@pytest.mark.integration
def test_run_experiment_minimal(tmp_path: Path) -> None:
    """Run a tiny experiment (2 samples, random weights, CPU) and verify artifacts."""
    cfg = OmegaConf.create({
        "task": {"name": "test", "seed": 0, "device": "cpu"},
        "dataset": {
            "name": "synthetic",
            "root": "",
            "num_samples": 2,
            "image_size": 32,
        },
        "target_model": {
            "type": "classifier",
            "name": "resnet50",
            "weights": "none",
        },
        "attacks": [{"name": "fgsm", "eps": 0.1}],
        "defenses": [{"name": "jpeg", "quality": 50}],
        "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
        "report": {
            "output_dir": str(tmp_path / "reports" / "test"),
            "formats": ["markdown"],
        },
    })

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)

    output_dir = run_experiment(config_path)

    # Verify output directory structure
    assert output_dir.exists()
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "metrics_wide.csv").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "samples").is_dir()
    assert (output_dir / "samples" / "clean").is_dir()
    assert (output_dir / "samples" / "adversarial").is_dir()
    assert (output_dir / "samples" / "purified").is_dir()
    assert (output_dir / "figures" / "sample_grid.png").exists()


@pytest.mark.integration
def test_run_experiment_multiple_attacks_defenses(tmp_path: Path) -> None:
    """Run with two attacks and two defenses, verify all metric keys appear."""
    cfg = OmegaConf.create({
        "task": {"name": "multi", "seed": 42, "device": "cpu"},
        "dataset": {
            "name": "synthetic",
            "root": "",
            "num_samples": 2,
            "image_size": 32,
        },
        "target_model": {
            "type": "classifier",
            "name": "resnet50",
            "weights": "none",
        },
        "attacks": [
            {"name": "fgsm", "eps": 0.1},
            {"name": "pgd", "eps": 0.1, "alpha": 0.02, "steps": 5},
        ],
        "defenses": [
            {"name": "jpeg", "quality": 50},
            {"name": "bit_depth", "bits": 4},
        ],
        "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
        "report": {
            "output_dir": str(tmp_path / "reports" / "multi"),
            "formats": ["markdown"],
        },
    })

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)

    output_dir = run_experiment(config_path)

    import json
    metrics = json.loads((output_dir / "metrics.json").read_text())

    # Check attack-level metrics exist
    assert "fgsm_asr" in metrics
    assert "pgd_asr" in metrics

    # Check defense-level metrics exist for each attack-defense pair
    assert "fgsm_vs_jpeg_robust_accuracy" in metrics
    assert "fgsm_vs_bit_depth_robust_accuracy" in metrics
    assert "pgd_vs_jpeg_robust_accuracy" in metrics
    assert "pgd_vs_bit_depth_robust_accuracy" in metrics


@pytest.mark.integration
def test_run_experiment_unknown_dataset_raises(tmp_path: Path) -> None:
    """Requesting an unsupported dataset should raise ValueError."""
    cfg = OmegaConf.create({
        "task": {"name": "bad", "seed": 0, "device": "cpu"},
        "dataset": {
            "name": "nonexistent_dataset",
            "root": str(tmp_path / "data"),
            "num_samples": 2,
            "image_size": 32,
        },
        "target_model": {
            "type": "classifier",
            "name": "resnet50",
            "weights": "none",
        },
        "attacks": [{"name": "fgsm", "eps": 0.1}],
        "defenses": [{"name": "jpeg", "quality": 50}],
        "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
        "report": {
            "output_dir": str(tmp_path / "reports" / "bad"),
            "formats": ["markdown"],
        },
    })

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)

    with pytest.raises(ValueError, match="Unknown dataset"):
        run_experiment(config_path)


@pytest.mark.integration
def test_run_experiment_with_quality_metrics(tmp_path: Path) -> None:
    """Test task runner with LPIPS quality metric."""
    cfg = OmegaConf.create({
        "task": {"name": "test_quality", "seed": 0, "device": "cpu"},
        "dataset": {
            "name": "synthetic",
            "root": "",
            "num_samples": 2,
            "image_size": 32,
        },
        "target_model": {
            "type": "classifier",
            "name": "resnet50",
            "weights": "none",
        },
        "attacks": [{"name": "fgsm", "eps": 0.1}],
        "defenses": [{"name": "jpeg", "quality": 50}],
        "metrics": {
            "attack": ["asr"],
            "defense": ["robust_accuracy"],
            "quality": ["lpips"],
        },
        "report": {
            "output_dir": str(tmp_path / "reports" / "test_quality"),
            "formats": ["markdown"],
        },
    })

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)

    import json

    output_dir = run_experiment(config_path)
    metrics = json.loads((output_dir / "metrics.json").read_text())

    assert "fgsm_lpips" in metrics
    assert isinstance(metrics["fgsm_lpips"], float)
    assert (output_dir / "figures" / "sample_grid.png").exists()


@pytest.mark.integration
def test_run_experiment_figures_directory_created(tmp_path: Path) -> None:
    """Verify figures directory is created even when no quality metrics are configured."""
    cfg = OmegaConf.create({
        "task": {"name": "test_figures", "seed": 0, "device": "cpu"},
        "dataset": {
            "name": "synthetic",
            "root": "",
            "num_samples": 2,
            "image_size": 32,
        },
        "target_model": {
            "type": "classifier",
            "name": "resnet50",
            "weights": "none",
        },
        "attacks": [{"name": "fgsm", "eps": 0.1}],
        "defenses": [{"name": "jpeg", "quality": 50}],
        "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
        "report": {
            "output_dir": str(tmp_path / "reports" / "test_figures"),
            "formats": ["markdown"],
        },
    })

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)

    output_dir = run_experiment(config_path)
    assert (output_dir / "figures").is_dir()
    assert (output_dir / "figures" / "sample_grid.png").exists()
    assert (output_dir / "report.md").exists()


@pytest.mark.integration
def test_clean_accuracy_drop_uses_clean_defended_samples(tmp_path: Path) -> None:
    """Clean accuracy drop should compare clean vs defended-clean samples."""
    cfg = OmegaConf.create({
        "task": {"name": "clean_drop", "seed": 0, "device": "cpu"},
        "dataset": {
            "name": "synthetic",
            "root": "",
            "num_samples": 2,
            "image_size": 32,
        },
        "target_model": {
            "type": "classifier",
            "name": "resnet50",
            "weights": "none",
        },
        "attacks": [{"name": "fgsm", "eps": 0.1}],
        "defenses": [{"name": "bit_depth", "bits": 4}],
        "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
        "report": {
            "output_dir": str(tmp_path / "reports" / "clean_drop"),
            "formats": ["markdown"],
        },
    })

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)

    import json

    output_dir = run_experiment(config_path)
    metrics = json.loads((output_dir / "metrics.json").read_text())

    clean_acc = metrics["fgsm_clean_accuracy"]
    clean_defended_acc = metrics["fgsm_vs_bit_depth_clean_defended_accuracy"]
    clean_drop = metrics["fgsm_vs_bit_depth_clean_accuracy_drop"]

    assert clean_drop == pytest.approx(clean_acc - clean_defended_acc)


def test_imagenet_subset_manifest_preserves_class_indices(tmp_path: Path) -> None:
    """ImageNet subset labels must come from the manifest, not folder order."""
    image_root = tmp_path / "imagenet_subset" / "images"
    image_dir = image_root / "n99999999"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "sample.JPEG"
    Image.new("RGB", (16, 16), color=(128, 64, 32)).save(image_path)

    manifest = tmp_path / "imagenet_subset" / "manifest.csv"
    manifest.write_text("path,label\nn99999999/sample.JPEG,407\n")

    cfg = OmegaConf.create({
        "name": "imagenet_subset",
        "root": str(image_root),
        "manifest": str(manifest),
        "num_samples": 1,
        "image_size": 16,
    })

    dataset = _load_dataset(cfg)
    _, label = dataset[0]
    assert label == 407
