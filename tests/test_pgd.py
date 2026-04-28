"""Tests for PGD attack implementation."""

import torch
import torch.nn as nn

from src.attack_engine.fgsm import FGSMAttack
from src.attack_engine.pgd import PGDAttack


class TinyClassifier(nn.Module):
    """Minimal classifier for testing without real weights.

    Uses a linear layer so every input pixel directly affects the output,
    making gradient-based perturbation maximally effective.
    """

    def __init__(self, img_size: int = 32) -> None:
        super().__init__()
        self.img_size = img_size
        self.fc = nn.Linear(3 * img_size * img_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class TestPGDOutputShape:
    def test_output_matches_input_shape(self) -> None:
        attack = PGDAttack()
        model = TinyClassifier()
        batch = torch.rand(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        result = attack.generate(batch, labels, model, config={})

        assert result.adversarial.shape == batch.shape


class TestPGDQueries:
    def test_all_queries_equal_steps(self) -> None:
        attack = PGDAttack()
        model = TinyClassifier()
        batch = torch.rand(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        result = attack.generate(batch, labels, model, config={"steps": 20})

        assert result.queries == [20, 20, 20, 20]


class TestPGDOutputRange:
    def test_output_clamped_to_unit_range(self) -> None:
        attack = PGDAttack()
        model = TinyClassifier()
        batch = torch.rand(2, 3, 32, 32)
        labels = torch.randint(0, 10, (2,))

        result = attack.generate(batch, labels, model, config={})

        assert result.adversarial.min() >= 0.0
        assert result.adversarial.max() <= 1.0


class TestPGDASRHigherThanFGSM:
    def test_pgd_asr_at_least_as_high_as_fgsm(self) -> None:
        """PGD (iterative) should have higher or equal ASR than FGSM (single-step).

        Use alpha large enough that PGD can traverse the full eps-ball
        within the allocated steps (alpha * steps >= eps).
        """
        torch.manual_seed(42)
        pgd = PGDAttack()
        fgsm = FGSMAttack()
        model = TinyClassifier()
        batch = torch.rand(32, 3, 32, 32)
        labels = torch.randint(0, 10, (32,))
        eps = 0.3
        steps = 20

        pgd_result = pgd.generate(
            batch, labels, model,
            config={"eps": eps, "alpha": eps * 2 / steps, "steps": steps},
        )
        fgsm_result = fgsm.generate(batch, labels, model, config={"eps": eps})

        pgd_asr = pgd_result.success.float().mean().item()
        fgsm_asr = fgsm_result.success.float().mean().item()

        assert pgd_asr >= fgsm_asr, (
            f"PGD ASR ({pgd_asr:.3f}) should be >= FGSM ASR ({fgsm_asr:.3f})"
        )


class TestPGDEpsEffect:
    def test_larger_eps_produces_larger_perturbation(self) -> None:
        attack = PGDAttack()
        model = TinyClassifier()
        batch = torch.rand(2, 3, 32, 32)
        labels = torch.randint(0, 10, (2,))

        small = attack.generate(batch, labels, model, config={"eps": 0.01})
        large = attack.generate(batch, labels, model, config={"eps": 0.3})

        delta_small = (small.adversarial - batch).abs().max().item()
        delta_large = (large.adversarial - batch).abs().max().item()

        assert delta_large > delta_small


class TestPGDStepsEffect:
    def test_more_steps_can_increase_asr(self) -> None:
        """More iterations give the attack more chances to find adversarial examples."""
        attack = PGDAttack()
        model = TinyClassifier()
        batch = torch.rand(32, 3, 32, 32)
        labels = torch.randint(0, 10, (32,))

        few_steps = attack.generate(batch, labels, model, config={"steps": 1})
        many_steps = attack.generate(batch, labels, model, config={"steps": 20})

        asr_few = few_steps.success.float().mean().item()
        asr_many = many_steps.success.float().mean().item()

        assert asr_many >= asr_few, (
            f"ASR with 20 steps ({asr_many:.3f}) should be >= ASR with 1 step ({asr_few:.3f})"
        )


class TestPGDMetadata:
    def test_metadata_contains_attack_params(self) -> None:
        attack = PGDAttack()
        model = TinyClassifier()
        batch = torch.rand(2, 3, 32, 32)
        labels = torch.randint(0, 10, (2,))

        result = attack.generate(
            batch, labels, model, config={"eps": 0.05, "alpha": 0.01, "steps": 10}
        )

        assert result.metadata["eps"] == 0.05
        assert result.metadata["alpha"] == 0.01
        assert result.metadata["steps"] == 10
        assert result.success.dtype == torch.bool
        assert len(result.queries) == 2


class TestPGDName:
    def test_attack_name(self) -> None:
        assert PGDAttack().name == "pgd"
