"""Tests for src.model_zoo.generators."""

import inspect

import pytest


def test_load_sd_pipeline_function_exists():
    """load_sd_pipeline should be importable and have the expected signature."""
    from src.model_zoo.generators import load_sd_pipeline

    sig = inspect.signature(load_sd_pipeline)
    assert "model_id" in sig.parameters
    assert "device" in sig.parameters
    assert "enable_attention_slicing" in sig.parameters
    assert "torch_dtype" in sig.parameters


def test_load_sd_pipeline_import_error():
    """Should raise ImportError with a helpful message when diffusers is missing."""
    try:
        import diffusers  # noqa: F401

        pytest.skip("diffusers is installed; cannot test ImportError path")
    except ImportError:
        from src.model_zoo.generators import load_sd_pipeline

        with pytest.raises(ImportError, match="diffusers"):
            load_sd_pipeline()
