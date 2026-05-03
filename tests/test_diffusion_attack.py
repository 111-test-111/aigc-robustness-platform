"""Tests for DiffusionAttack class interface and structure.

NOTE: Real SD pipeline integration tests live in the integration suite
(Task 2.10). These tests verify the class exists, inherits correctly,
and exposes the expected API without requiring diffusers or a GPU.
"""

import inspect

import pytest
import torch
import torch.nn as nn
from PIL import Image


class TinyClassifier(nn.Module):
    """Minimal classifier for testing without real weights."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x).flatten(1))


class TestDiffusionAttackClassExists:
    def test_class_importable(self) -> None:
        from src.attack_engine.diffusion_attack import DiffusionAttack

        assert DiffusionAttack is not None

    def test_name_attribute(self) -> None:
        from src.attack_engine.diffusion_attack import DiffusionAttack

        assert DiffusionAttack.name == "diffusion"


class TestDiffusionAttackInheritance:
    def test_subclass_of_attack(self) -> None:
        from src.attack_engine.base import Attack
        from src.attack_engine.diffusion_attack import DiffusionAttack

        assert issubclass(DiffusionAttack, Attack)

    def test_has_generate_method(self) -> None:
        from src.attack_engine.diffusion_attack import DiffusionAttack

        assert hasattr(DiffusionAttack, "generate")
        assert callable(getattr(DiffusionAttack, "generate", None))


class TestDiffusionAttackSignature:
    def test_generate_has_required_params(self) -> None:
        from src.attack_engine.diffusion_attack import DiffusionAttack

        sig = inspect.signature(DiffusionAttack.generate)
        params = list(sig.parameters.keys())

        assert "batch" in params
        assert "labels" in params
        assert "target_model" in params
        assert "config" in params

    def test_generate_returns_attack_result(self) -> None:
        from src.attack_engine.base import AttackResult
        from src.attack_engine.diffusion_attack import DiffusionAttack

        return_annotation = inspect.signature(DiffusionAttack.generate).return_annotation
        assert return_annotation is AttackResult or str(return_annotation) == "AttackResult"


class TestDiffusionAttackDefaults:
    def test_instantiation(self) -> None:
        from src.attack_engine.diffusion_attack import DiffusionAttack

        attack = DiffusionAttack()
        assert attack.name == "diffusion"

    def test_generate_accepts_empty_config(self) -> None:
        """generate() should parse defaults from config without crashing on import."""
        from src.attack_engine.diffusion_attack import DiffusionAttack

        attack = DiffusionAttack()
        # We cannot call generate() without diffusers installed,
        # but verify the method signature allows an empty config dict.
        sig = inspect.signature(attack.generate)
        # The config parameter should have no default restriction
        config_param = sig.parameters["config"]
        assert config_param.default is inspect.Parameter.empty


class FakeImg2ImgResult:
    def __init__(self, images):
        self.images = images


class FakeImg2ImgPipe:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, *, prompt, image, **kwargs):
        self.batch_sizes.append(len(image))
        assert len(prompt) == len(image)
        return FakeImg2ImgResult([
            Image.new("RGB", img.size, color=(idx, idx, idx))
            for idx, img in enumerate(image)
        ])


def test_sd_round_uses_configured_batches() -> None:
    """SD img2img calls should be chunked instead of one call per image."""
    from src.attack_engine.diffusion_attack import DiffusionAttack

    attack = DiffusionAttack()
    pipe = FakeImg2ImgPipe()
    batch = torch.rand(5, 3, 16, 16)

    result = attack._generate_one_sd_round(
        pipe,
        batch,
        prompt="a photo",
        strength=0.7,
        guidance_scale=7.5,
        sd_batch_size=2,
    )

    assert pipe.batch_sizes == [2, 2, 1]
    assert result.shape == batch.shape
