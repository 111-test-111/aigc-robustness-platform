"""Tests for bit-depth reduction defense."""

import pytest
import torch

from src.defense_engine.bit_depth import BitDepthDefense


@pytest.fixture
def defense():
    return BitDepthDefense()


@pytest.fixture
def batch():
    torch.manual_seed(42)
    return torch.rand(4, 3, 32, 32)


def test_output_shape(defense, batch):
    result = defense.apply(batch, {})
    assert result.defended.shape == batch.shape


def test_output_range(defense, batch):
    result = defense.apply(batch, {})
    assert result.defended.min() >= 0.0
    assert result.defended.max() <= 1.0


def test_quantization_reduces_unique_values(defense):
    torch.manual_seed(0)
    batch = torch.rand(2, 3, 16, 16)

    result = defense.apply(batch, {"bits": 4})
    unique_original = batch.unique().numel()
    unique_defended = result.defended.unique().numel()

    assert unique_defended < unique_original


def test_bits_effect(defense):
    torch.manual_seed(0)
    batch = torch.rand(2, 3, 16, 16)

    result_4bit = defense.apply(batch, {"bits": 4})
    result_2bit = defense.apply(batch, {"bits": 2})

    unique_4bit = result_4bit.defended.unique().numel()
    unique_2bit = result_2bit.defended.unique().numel()

    assert unique_2bit < unique_4bit


def test_latency(defense, batch):
    result = defense.apply(batch, {})
    assert result.latency_sec > 0
