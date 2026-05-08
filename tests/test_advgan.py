"""Tests for AdvGAN generative adversarial attack implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attack_engine.adv_gan import AdvGANAttack


class TinyClassifier(nn.Module):
    """Minimal classifier for testing without real weights.

    Uses a linear layer so every input pixel directly affects the output,
    making perturbation maximally effective.
    """

    def __init__(self, img_size: int = 32) -> None:
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(3 * img_size * img_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class TestAdvGANOutputShape:
    def test_output_matches_input_shape(self) -> None:
        attack = AdvGANAttack()
        model = TinyClassifier()
        batch = torch.rand(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        result = attack.generate(batch, labels, model, config={})

        assert result.adversarial.shape == batch.shape


class TestAdvGANOutputRange:
    def test_output_clamped_to_unit_range(self) -> None:
        attack = AdvGANAttack()
        model = TinyClassifier()
        batch = torch.rand(2, 3, 32, 32)
        labels = torch.randint(0, 10, (2,))

        result = attack.generate(batch, labels, model, config={})

        assert result.adversarial.min() >= 0.0
        assert result.adversarial.max() <= 1.0


class TestAdvGANSuccessMask:
    def test_success_is_boolean_tensor(self) -> None:
        attack = AdvGANAttack()
        model = TinyClassifier()
        batch = torch.rand(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        result = attack.generate(batch, labels, model, config={})

        assert result.success.dtype == torch.bool
        assert result.success.shape == (4,)


class TestAdvGANMetadata:
    def test_metadata_contains_params(self) -> None:
        attack = AdvGANAttack()
        model = TinyClassifier()
        batch = torch.rand(2, 3, 32, 32)
        labels = torch.randint(0, 10, (2,))

        result = attack.generate(
            batch, labels, model, config={"eps": 0.05, "epochs": 30}
        )

        assert result.metadata["eps"] == 0.05
        assert result.metadata["epochs"] == 30
        assert "lr" in result.metadata
        assert "elapsed_sec" in result.metadata


class TestAdvGANName:
    def test_attack_name(self) -> None:
        assert AdvGANAttack().name == "advgan"


class TestAdvGANQueries:
    def test_queries_equal_epochs(self) -> None:
        attack = AdvGANAttack()
        model = TinyClassifier()
        batch = torch.rand(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        result = attack.generate(batch, labels, model, config={"epochs": 20})

        assert result.queries == [20, 20, 20, 20]


class TestAdvGANObjective:
    def test_untargeted_loss_maximizes_true_label_cross_entropy(self) -> None:
        logits = torch.tensor([[4.0, 1.0], [0.5, 3.0]])
        labels = torch.tensor([0, 1])

        loss, objective = AdvGANAttack._classification_loss(logits, labels, None)

        assert objective == "untargeted_ce_ascent"
        assert torch.allclose(loss, -F.cross_entropy(logits, labels))

    def test_targeted_loss_minimizes_target_label_cross_entropy(self) -> None:
        logits = torch.tensor([[4.0, 1.0], [0.5, 3.0]])
        labels = torch.tensor([0, 1])

        loss, objective = AdvGANAttack._classification_loss(logits, labels, 0)

        assert objective == "targeted_ce"
        assert torch.allclose(loss, F.cross_entropy(logits, torch.zeros_like(labels)))


class TestAdvGANEpsEffect:
    def test_larger_eps_produces_larger_perturbation(self) -> None:
        torch.manual_seed(42)
        attack = AdvGANAttack()
        model = TinyClassifier()
        batch = torch.rand(2, 3, 32, 32)
        labels = torch.randint(0, 10, (2,))

        small = attack.generate(batch, labels, model, config={"eps": 0.01, "epochs": 50})
        large = attack.generate(batch, labels, model, config={"eps": 0.3, "epochs": 50})

        delta_small = (small.adversarial - batch).abs().max().item()
        delta_large = (large.adversarial - batch).abs().max().item()

        assert delta_large > delta_small
