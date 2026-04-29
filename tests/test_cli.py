"""CLI entrypoint tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.mark.integration
def test_python_m_src_cli_run_executes(tmp_path: Path) -> None:
    """The documented ``python -m src.cli run`` entrypoint should run."""
    cfg = OmegaConf.create({
        "task": {"name": "cli_smoke", "seed": 0, "device": "cpu"},
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
        "attacks": [{"name": "fgsm", "eps": 0.05}],
        "defenses": [{"name": "jpeg", "quality": 75}],
        "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
        "report": {
            "output_dir": str(tmp_path / "reports" / "cli_smoke"),
            "formats": ["markdown"],
        },
    })

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(cfg, config_path)

    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "run", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "实验完成" in result.stdout
    assert (tmp_path / "reports" / "cli_smoke" / "report.md").exists()
