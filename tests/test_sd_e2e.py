"""Slow end-to-end test for the real diffusers Stable Diffusion backend."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.task_runner import run_experiment


@pytest.mark.slow
@pytest.mark.e2e
def test_real_stable_diffusion_backend_e2e(tmp_path: Path) -> None:
    """Run a one-sample experiment through ``backend: sd``.

    This test is opt-in because it may download model weights and can be slow.
    Set ``RUN_SLOW_SD_E2E=1`` to enable it. By default it uses Hugging Face's
    tiny SD pipeline; set ``SD_E2E_MODEL_ID`` to a full local or remote model
    id when validating the final paper environment.
    """
    if os.environ.get("RUN_SLOW_SD_E2E") != "1":
        pytest.skip("set RUN_SLOW_SD_E2E=1 to run the real SD e2e test")

    pytest.importorskip("diffusers")

    model_id = os.environ.get(
        "SD_E2E_MODEL_ID", "hf-internal-testing/tiny-stable-diffusion-pipe"
    )

    cfg = OmegaConf.create({
        "task": {"name": "sd_e2e", "seed": 0, "device": "auto"},
        "dataset": {
            "name": "synthetic",
            "root": "",
            "num_samples": 1,
            "image_size": 64,
            "batch_size": 1,
        },
        "target_model": {
            "type": "classifier",
            "name": "resnet50",
            "weights": "none",
        },
        "attacks": [{
            "name": "diffusion",
            "backend": "sd",
            "generator": model_id,
            "prompt": "a high quality photo",
            "strength": 0.5,
            "guidance_scale": 1.0,
            "num_candidates": 1,
        }],
        "defenses": [{
            "name": "diffusion_purification",
            "backend": "sd",
            "model_id": model_id,
            "noise_level": 0.05,
            "steps": 2,
        }],
        "metrics": {"attack": ["asr"], "defense": ["robust_accuracy"]},
        "report": {
            "output_dir": str(tmp_path / "reports" / "sd_e2e"),
            "formats": ["markdown"],
        },
    })

    config_path = tmp_path / "sd_e2e.yaml"
    OmegaConf.save(cfg, config_path)

    output_dir = run_experiment(config_path)

    assert (output_dir / "report.md").exists()
    assert (output_dir / "samples" / "adversarial" / "diffusion").is_dir()

    structured = OmegaConf.load(output_dir / "structured_metrics.json")
    defense_info = structured.defenses.diffusion_vs_diffusion_purification
    assert defense_info.metadata.actual_backend == "stable_diffusion"
