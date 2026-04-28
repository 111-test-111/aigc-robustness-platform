"""Tests for FGSM attack implementation."""

import torch
import torch.nn as nn

from src.attack_engine.fgsm import FGSMAttack


class TinyClassifier(nn.Module):
    """Minimal classifier for testing without real weights.

    Uses a linear layer so every input pixel directly affects the output,
    making FGSM gradient perturbation maximally effective.
    """

    def __init__(self, img_size: int = 32) -> None:
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(3 * img_size * img_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class TestFGSMOutputShape:
    def test_output_matches_input_shape(self) -> None:
        attack = FGSMAttack()
        model = TinyClassifier()
        batch = torch.rand(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        result = attack.generate(batch, labels, model, config={})

        assert result.adversarial.shape == batch.shape


class TestFGSMQueries:
    def test_all_queries_are_one(self) -> None:
        attack = FGSMAttack()
        model = TinyClassifier()
        batch = torch.rand(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        result = attack.generate(batch, labels, model, config={})

        assert result.queries == [1, 1, 1, 1]


class TestFGSMOutputRange:
    def test_output_clamped_to_unit_range(self) -> None:
        attack = FGSMAttack()
        model = TinyClassifier()
        batch = torch.rand(2, 3, 32, 32)
        labels = torch.randint(0, 10, (2,))

        result = attack.generate(batch, labels, model, config={})

        assert result.adversarial.min() >= 0.0
        assert result.adversarial.max() <= 1.0


class TestFGSMNoiseMagnitude:
    def test_larger_eps_produces_larger_perturbation(self) -> None:
        attack = FGSMAttack()
        model = TinyClassifier()
        batch = torch.rand(2, 3, 32, 32)
        labels = torch.randint(0, 10, (2,))

        small = attack.generate(batch, labels, model, config={"eps": 0.01})
        large = attack.generate(batch, labels, model, config={"eps": 0.3})

        delta_small = (small.adversarial - batch).abs().max().item()
        delta_large = (large.adversarial - batch).abs().max().item()

        assert delta_large > delta_small


class TestFGSMSuccess:
    def test_at_least_one_sample_attacked_successfully(self) -> None:
        attack = FGSMAttack()
        model = TinyClassifier()
        # Large batch gives higher chance of at least one success
        batch = torch.rand(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))

        result = attack.generate(batch, labels, model, config={"eps": 0.5})

        assert result.success.any(), "FGSM should fool at least one sample with high eps"


class TestFGSMReturnTypes:
    def test_metadata_and_shape(self) -> None:
        attack = FGSMAttack()
        model = TinyClassifier()
        batch = torch.rand(3, 3, 32, 32)
        labels = torch.randint(0, 10, (3,))

        result = attack.generate(batch, labels, model, config={})

        assert result.metadata["eps"] == 0.03
        assert result.success.dtype == torch.bool
        assert len(result.queries) == 3
