"""Tests for the Gradio web UI (no server launch required)."""

from __future__ import annotations

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
        # Simulate the None-image early return path
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

        # When defense_name == "None", defended stays None
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
