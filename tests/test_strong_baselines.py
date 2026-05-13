"""Tests for strong-baseline attack adapters."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.attack_engine.autoattack_adapter import AutoAttackAdapter
from src.attack_engine.external_command import AdvDiffExternalAttack, AdvDiffuserExternalAttack


class MeanClassifier(nn.Module):
    """Tiny deterministic classifier for adapter tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = x.mean(dim=(1, 2, 3))
        return torch.stack([1.0 - score, score], dim=1)


def test_autoattack_adapter_uses_external_package(monkeypatch) -> None:
    class FakeAutoAttack:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            self.seed = kwargs.get("seed")
            self.attacks_to_run = []

        def run_standard_evaluation(self, images, labels, bs):
            assert bs == 2
            assert self.kwargs["norm"] == "Linf"
            assert self.kwargs["eps"] == 0.1
            return torch.clamp(images + 0.5, 0, 1)

    fake_module = types.ModuleType("autoattack")
    fake_module.AutoAttack = FakeAutoAttack
    monkeypatch.setitem(sys.modules, "autoattack", fake_module)

    attack = AutoAttackAdapter()
    batch = torch.zeros(2, 3, 8, 8)
    labels = torch.zeros(2, dtype=torch.long)

    result = attack.generate(
        batch,
        labels,
        MeanClassifier(),
        {"eps": 0.1, "batch_size": 2, "seed": 123},
    )

    assert result.adversarial.shape == batch.shape
    assert result.success.dtype == torch.bool
    assert result.queries == [5300, 5300]
    assert result.metadata["backend"] == "autoattack"
    assert result.metadata["queries_are_estimated"] is True


def test_autoattack_adapter_falls_back_to_pyautoattack(monkeypatch) -> None:
    class FakePyAutoAttack:
        def __init__(self, model, *, norm, eps, seed, version, device):
            self.model = model
            self.kwargs = {
                "norm": norm,
                "eps": eps,
                "seed": seed,
                "version": version,
                "device": device,
            }
            self.attacks_to_run = []

        def run_standard_evaluation(self, images, labels, *, batch_size):
            assert batch_size == 2
            assert self.kwargs["norm"] == "Linf"
            assert self.kwargs["eps"] == 0.1
            return torch.clamp(images + 0.5, 0, 1)

    fake_module = types.ModuleType("pyautoattack")
    fake_module.AutoAttack = FakePyAutoAttack
    monkeypatch.setitem(sys.modules, "pyautoattack", fake_module)
    monkeypatch.delitem(sys.modules, "autoattack", raising=False)

    attack = AutoAttackAdapter()
    batch = torch.zeros(2, 3, 8, 8)
    labels = torch.zeros(2, dtype=torch.long)

    result = attack.generate(
        batch,
        labels,
        MeanClassifier(),
        {"eps": 0.1, "batch_size": 2, "seed": 123, "verbose": True, "log_path": "ignored.log"},
    )

    assert result.adversarial.shape == batch.shape
    assert result.metadata["backend"] == "pyautoattack"


def test_autoattack_rejects_targeted_config(monkeypatch) -> None:
    attack = AutoAttackAdapter()
    batch = torch.zeros(1, 3, 8, 8)
    labels = torch.zeros(1, dtype=torch.long)

    with pytest.raises(ValueError, match="untargeted"):
        attack.generate(batch, labels, MeanClassifier(), {"target_class": 1})


def test_advdiffuser_external_command_roundtrip(tmp_path: Path) -> None:
    script_path = tmp_path / "external_attack.py"
    script_path.write_text(
        """
import argparse
import csv
import json
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--metadata", required=True)
args = parser.parse_args()

with open(args.manifest, newline="") as f:
    for row in csv.DictReader(f):
        image = Image.open(row["input_path"]).convert("RGB")
        image.save(row["output_path"])

with open(args.metadata, "w") as f:
    json.dump({"queries_per_sample": 7, "source": "test-script"}, f)
""".strip()
    )

    attack = AdvDiffuserExternalAttack()
    batch = torch.rand(2, 3, 8, 8)
    labels = torch.zeros(2, dtype=torch.long)

    result = attack.generate(
        batch,
        labels,
        MeanClassifier(),
        {
            "command": [
                sys.executable,
                str(script_path),
                "--manifest",
                "{manifest}",
                "--metadata",
                "{metadata_json}",
            ],
            "timeout_sec": 10,
        },
    )

    assert result.adversarial.shape == batch.shape
    assert result.success.shape == (2,)
    assert result.queries == [7, 7]
    assert result.metadata["method"] == "AdvDiffuser"
    assert result.metadata["external_metadata"]["source"] == "test-script"


def test_external_command_requires_command() -> None:
    attack = AdvDiffExternalAttack()
    batch = torch.rand(1, 3, 8, 8)
    labels = torch.zeros(1, dtype=torch.long)

    with pytest.raises(ValueError, match="requires a `command`"):
        attack.generate(batch, labels, MeanClassifier(), {})


def test_strong_baseline_registry_and_configs_load() -> None:
    from src.task_runner import ATTACK_REGISTRY

    assert "autoattack" in ATTACK_REGISTRY
    assert "advdiffuser" in ATTACK_REGISTRY
    assert "advdiff" in ATTACK_REGISTRY
    assert "generator_baseline" in ATTACK_REGISTRY

    autoattack_cfg = OmegaConf.load("configs/paper/06_strong_linf_autoattack.yaml")
    assert autoattack_cfg.attacks[0].name == "autoattack"

    advdiff_paper_cfg = OmegaConf.load("configs/paper/07_advdiff_external.yaml")
    assert advdiff_paper_cfg.attacks[0].name == "advdiff"
    assert advdiff_paper_cfg.dataset.num_samples == 200
    assert advdiff_paper_cfg.dataset.eval_batch_size == 200

    advdiffuser_cfg = OmegaConf.load("configs/templates/advdiffuser_external.yaml")
    assert advdiffuser_cfg.attacks[0].name == "advdiffuser"

    advdiff_cfg = OmegaConf.load("configs/templates/advdiff_external.yaml")
    assert advdiff_cfg.attacks[0].name == "advdiff"


def test_external_diffusion_baselines_are_scheduled_as_sd_heavy() -> None:
    from src.cli import _is_sd_config

    assert _is_sd_config(Path("configs/templates/advdiffuser_external.yaml"))
    assert _is_sd_config(Path("configs/templates/advdiff_external.yaml"))
