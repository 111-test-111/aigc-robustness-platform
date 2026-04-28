import inspect
import time

import pytest
import torch

from src.defense_engine.base import Defense, DefenseResult


class ConcreteDefense(Defense):
    """Minimal concrete subclass for testing."""

    name = "concrete_test"

    def apply(self, batch, config):
        start = time.perf_counter()
        defended = batch.clone()
        latency = time.perf_counter() - start
        return DefenseResult(defended=defended, latency_sec=latency)


class TestDefenseResult:
    def test_defense_result_fields(self):
        defended = torch.rand(2, 3, 32, 32)
        latency = 0.05

        result = DefenseResult(defended=defended, latency_sec=latency)

        assert result.defended.shape == (2, 3, 32, 32)
        assert result.latency_sec == 0.05

    def test_defense_result_tensor_is_accessible(self):
        original = torch.rand(1, 3, 8, 8)
        result = DefenseResult(defended=original, latency_sec=0.01)

        assert result.defended is original


class TestDefenseSubclass:
    def test_defense_subclass(self):
        defense = ConcreteDefense()
        assert defense.name == "concrete_test"
        assert isinstance(defense, Defense)

    def test_defense_apply_signature(self):
        sig = inspect.signature(ConcreteDefense.apply)
        params = list(sig.parameters.keys())
        assert params == ["self", "batch", "config"]

    def test_concrete_apply_returns_defense_result(self):
        defense = ConcreteDefense()
        batch = torch.rand(4, 3, 32, 32)

        result = defense.apply(batch, config={})

        assert isinstance(result, DefenseResult)
        assert result.defended.shape == (4, 3, 32, 32)
        assert result.latency_sec >= 0
