import inspect
import pytest
import torch

from src.attack_engine.base import Attack, AttackResult


class ConcreteAttack(Attack):
    """Minimal concrete subclass for testing."""

    name = "concrete_test"

    def generate(self, batch, labels, target_model, config):
        b = batch.shape[0]
        return AttackResult(
            adversarial=batch.clone(),
            success=torch.zeros(b, dtype=torch.bool),
            queries=[0] * b,
        )


class TestAttackResult:
    def test_attack_result_fields(self):
        adv = torch.randn(2, 3, 32, 32)
        success = torch.tensor([True, False])
        queries = [10, 20]
        metadata = {"elapsed_sec": 1.5}

        result = AttackResult(
            adversarial=adv,
            success=success,
            queries=queries,
            metadata=metadata,
        )

        assert result.adversarial.shape == (2, 3, 32, 32)
        assert result.success.shape == (2,)
        assert result.success[0] is True or result.success[0].item() is True
        assert result.queries == [10, 20]
        assert result.metadata["elapsed_sec"] == 1.5

    def test_attack_result_default_metadata(self):
        result = AttackResult(
            adversarial=torch.randn(1, 3, 8, 8),
            success=torch.tensor([True]),
            queries=[5],
        )

        assert result.metadata == {}


class TestAttackSubclass:
    def test_attack_subclass(self):
        attack = ConcreteAttack()
        assert attack.name == "concrete_test"
        assert isinstance(attack, Attack)

    def test_attack_generate_signature(self):
        sig = inspect.signature(ConcreteAttack.generate)
        params = list(sig.parameters.keys())
        assert params == ["self", "batch", "labels", "target_model", "config"]

    def test_concrete_generate_returns_attack_result(self):
        attack = ConcreteAttack()
        batch = torch.rand(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        model = torch.nn.Linear(10, 10)  # dummy model

        result = attack.generate(batch, labels, model, config={})

        assert isinstance(result, AttackResult)
        assert result.adversarial.shape == (4, 3, 32, 32)
        assert result.success.shape == (4,)
        assert len(result.queries) == 4
