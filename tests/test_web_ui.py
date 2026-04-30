"""Tests for the Gradio web UI (no server launch required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torchvision.transforms as T
from PIL import Image

gradio = pytest.importorskip("gradio")

from src.attack_engine.fgsm import FGSMAttack  # noqa: E402
from src.defense_engine.bit_depth import BitDepthDefense  # noqa: E402
from src.defense_engine.blur import GaussianBlurDefense  # noqa: E402
from src.defense_engine.jpeg import JPEGDefense  # noqa: E402
from src.model_zoo.classifiers import load_classifier  # noqa: E402
from src.web_ui import create_app  # noqa: E402


@pytest.fixture()
def sample_image() -> Image.Image:
    """Create a random 224x224 RGB image for testing."""
    return Image.new("RGB", (224, 224), color=(128, 64, 200))


@pytest.fixture()
def model() -> torch.nn.Module:
    """Load the ResNet50 classifier for testing."""
    return load_classifier("resnet50", "imagenet", torch.device("cpu"))


@pytest.fixture()
def image_tensor(sample_image: Image.Image) -> torch.Tensor:
    """Convert sample image to a (1, 3, 224, 224) tensor."""
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    return transform(sample_image).unsqueeze(0)


class TestCreateApp:
    """Tests for create_app."""

    def test_returns_gradio_blocks(self) -> None:
        """create_app should return a gr.Blocks instance."""
        app = create_app("resnet50", "cpu")
        assert isinstance(app, gradio.Blocks)

    def test_app_is_not_none(self) -> None:
        """The Blocks app should be constructable without errors."""
        app = create_app("resnet50", "cpu")
        assert app is not None


class TestProcessLogic:
    """Tests for the internal process logic used by the UI."""

    def test_returns_five_outputs(self, image_tensor: torch.Tensor, model: torch.nn.Module) -> None:
        """The process function should return exactly 5 outputs."""
        to_pil = T.ToPILImage()

        with torch.no_grad():
            labels = torch.tensor([model(image_tensor).argmax(dim=1).item()])

        attack = FGSMAttack()
        result = attack.generate(image_tensor, labels, model, {"eps": 0.03})

        orig_img = to_pil(image_tensor[0].clamp(0, 1))
        adv_img = to_pil(result.adversarial[0].clamp(0, 1))
        delta = (result.adversarial - image_tensor).abs().max().item()
        success = "Yes" if result.success[0].item() else "No"

        outputs = (orig_img, adv_img, None, f"L-inf: {delta:.4f}", f"Attack Success: {success}")
        assert len(outputs) == 5

    def test_none_image_returns_nones(self) -> None:
        """When no image is provided, all outputs should be None with info text."""
        result = (None, None, None, "No image provided", "No image provided")
        assert result[0] is None
        assert result[1] is None
        assert result[2] is None
        assert "No image provided" in result[3]

    def test_without_defense_defended_is_none(self, image_tensor: torch.Tensor, model: torch.nn.Module) -> None:
        """When defense is 'None', defended image should be None."""
        with torch.no_grad():
            labels = torch.tensor([model(image_tensor).argmax(dim=1).item()])

        attack = FGSMAttack()
        result = attack.generate(image_tensor, labels, model, {"eps": 0.03})

        defended = None
        to_pil = T.ToPILImage()
        def_img = to_pil(defended[0].clamp(0, 1)) if defended is not None else None
        assert def_img is None

    def test_with_jpeg_defense(self, image_tensor: torch.Tensor, model: torch.nn.Module) -> None:
        """JPEG defense should produce a valid defended tensor with correct shape."""
        with torch.no_grad():
            labels = torch.tensor([model(image_tensor).argmax(dim=1).item()])

        attack = FGSMAttack()
        result = attack.generate(image_tensor, labels, model, {"eps": 0.03})

        defense = JPEGDefense()
        d_result = defense.apply(result.adversarial, {"quality": 75})
        assert d_result.defended.shape == result.adversarial.shape
        assert d_result.defended.min() >= 0.0
        assert d_result.defended.max() <= 1.0

    def test_with_blur_defense(self, image_tensor: torch.Tensor, model: torch.nn.Module) -> None:
        """Gaussian blur defense should produce a valid defended tensor."""
        with torch.no_grad():
            labels = torch.tensor([model(image_tensor).argmax(dim=1).item()])

        attack = FGSMAttack()
        result = attack.generate(image_tensor, labels, model, {"eps": 0.03})

        defense = GaussianBlurDefense()
        d_result = defense.apply(result.adversarial, {})
        assert d_result.defended.shape == result.adversarial.shape

    def test_with_bit_depth_defense(self, image_tensor: torch.Tensor, model: torch.nn.Module) -> None:
        """Bit depth defense should produce a valid defended tensor."""
        with torch.no_grad():
            labels = torch.tensor([model(image_tensor).argmax(dim=1).item()])

        attack = FGSMAttack()
        result = attack.generate(image_tensor, labels, model, {"eps": 0.03})

        defense = BitDepthDefense()
        d_result = defense.apply(result.adversarial, {})
        assert d_result.defended.shape == result.adversarial.shape

    def test_perturbation_metric(self, image_tensor: torch.Tensor, model: torch.nn.Module) -> None:
        """Perturbation metric should be a non-negative L-inf value."""
        with torch.no_grad():
            labels = torch.tensor([model(image_tensor).argmax(dim=1).item()])

        attack = FGSMAttack()
        result = attack.generate(image_tensor, labels, model, {"eps": 0.03})

        delta = (result.adversarial - image_tensor).abs().max().item()
        assert delta >= 0.0
        assert isinstance(delta, float)


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tests for the web_ui helper functions."""

    def test_metrics_to_table(self) -> None:
        from src.web_ui import _metrics_to_table

        metrics = {"asr": 0.95, "accuracy": 0.83, "name": "test"}
        rows = _metrics_to_table(metrics)
        assert len(rows) == 3
        # Should be sorted by key
        assert rows[0][0] == "accuracy"
        assert rows[0][1] == "0.8300"
        assert rows[1][0] == "asr"
        assert rows[2][0] == "name"

    def test_structured_to_attack_table(self) -> None:
        from src.web_ui import _structured_to_attack_table

        structured = {
            "attacks": {
                "fgsm": {
                    "asr": 0.35,
                    "asr_on_clean_correct": 0.36,
                    "clean_accuracy": 0.83,
                    "adversarial_accuracy": 0.53,
                    "prediction_change_rate": 0.35,
                },
            },
        }
        rows = _structured_to_attack_table(structured)
        assert len(rows) == 1
        assert rows[0][0] == "fgsm"
        assert "35.0%" in rows[0][1]

    def test_structured_to_defense_table(self) -> None:
        from src.web_ui import _structured_to_defense_table

        structured = {
            "defenses": {
                "fgsm_vs_jpeg": {
                    "robust_accuracy": 0.59,
                    "clean_accuracy_drop": 0.07,
                    "clean_defended_accuracy": 0.76,
                    "latency": {"mean": 0.095},
                },
            },
        }
        rows = _structured_to_defense_table(structured)
        assert len(rows) == 1
        assert rows[0][0] == "fgsm_vs_jpeg"
        assert "59.0%" in rows[0][1]

    def test_list_experiment_configs(self) -> None:
        from src.web_ui import _list_experiment_configs

        configs = _list_experiment_configs()
        names = [c.stem for c in configs]
        # Should include the canonical paper experiments
        assert "01_traditional_attack_baseline" in names
        assert "02_generative_attack_mainline" in names
        assert "03_full_resnet50_suite" in names
        # Should include ablations
        assert "00_no_defense" in names
        assert "strength_03" in names
        assert "steps_10" in names
        # Should NOT include smoke configs
        assert "cpu" not in names
        assert "synthetic" not in names
        assert "sd_tiny_e2e" not in names

    def test_list_completed_experiments(self) -> None:
        from src.web_ui import _list_completed_experiments

        names = _list_completed_experiments()
        # baseline_resnet50 was just run, should exist
        assert "baseline_resnet50" in names

    def test_load_experiment_metrics(self) -> None:
        from src.web_ui import _load_experiment_metrics

        metrics = _load_experiment_metrics("baseline_resnet50")
        assert "attacks" in metrics
        assert "defenses" in metrics
        assert "fgsm" in metrics["attacks"]

    def test_load_flat_metrics(self) -> None:
        from src.web_ui import _load_flat_metrics

        metrics = _load_flat_metrics("baseline_resnet50")
        assert "fgsm_asr" in metrics
        assert isinstance(metrics["fgsm_asr"], (int, float))

    def test_load_report_markdown(self) -> None:
        from src.web_ui import _load_report_markdown

        report = _load_report_markdown("baseline_resnet50")
        assert "# 实验报告" in report
        assert "baseline_resnet50" in report

    def test_load_report_markdown_missing(self) -> None:
        from src.web_ui import _load_report_markdown

        report = _load_report_markdown("nonexistent_experiment")
        assert "No report found" in report

    def test_list_figures(self) -> None:
        from src.web_ui import _list_figures

        figures = _list_figures("baseline_resnet50")
        assert len(figures) > 0
        assert any("radar" in f.name for f in figures)

    def test_list_sample_dirs(self) -> None:
        from src.web_ui import _list_sample_dirs

        dirs = _list_sample_dirs("baseline_resnet50")
        assert any("clean" in d for d in dirs)
        assert any("adversarial" in d for d in dirs)

    def test_load_sample_images(self) -> None:
        from src.web_ui import _load_sample_images

        # Test with "samples/" prefix (as returned by _list_sample_dirs)
        images = _load_sample_images("baseline_resnet50", "samples/clean")
        assert len(images) > 0
        assert all(p.suffix == ".png" for p in images)

    def test_load_sample_images_empty(self) -> None:
        from src.web_ui import _load_sample_images

        images = _load_sample_images("baseline_resnet50", "nonexistent_subdir")
        assert images == []
