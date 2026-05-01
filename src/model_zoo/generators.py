"""Generator model loaders for diffusion pipelines."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Process-level cache for SD pipelines to avoid reloading
_PIPELINE_CACHE: dict[tuple[str, str, str], Any] = {}


def load_sd_pipeline(
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    device: torch.device = torch.device("cpu"),
    enable_attention_slicing: bool = True,
    torch_dtype: torch.dtype = torch.float16,
    disable_safety_checker: bool = True,
) -> Any:
    """Load a Stable Diffusion img2img pipeline.

    Pipelines are cached by (model_id, device, dtype) to avoid repeated
    loading within the same process.

    Args:
        model_id: HuggingFace model identifier or local path.
        device: Target device for the pipeline.
        enable_attention_slicing: Reduce VRAM usage via attention slicing.
        torch_dtype: Precision for model weights.
        disable_safety_checker: Disable diffusers safety checker. The platform
            evaluates robustness on curated research datasets, and tiny testing
            pipelines often ship incompatible safety-checker image processors.

    Returns:
        An ``AutoPipelineForImage2Image`` instance on *device*.

    Raises:
        ImportError: If the ``diffusers`` package is not installed.
    """
    try:
        from diffusers import AutoPipelineForImage2Image
    except ImportError:
        raise ImportError(
            "diffusers is required. Install with: pip install diffusers"
        )

    if device.type == "cpu" and torch_dtype == torch.float16:
        logger.warning(
            "float16 SD pipeline on %s; switching to float32 for numerical stability",
            device.type,
        )
        torch_dtype = torch.float32

    # Check cache
    cache_key = (model_id, str(device), str(torch_dtype))
    if cache_key in _PIPELINE_CACHE:
        logger.debug("Using cached SD pipeline for %s", cache_key)
        return _PIPELINE_CACHE[cache_key]

    logger.info("Loading SD pipeline from %s (dtype=%s)", model_id, torch_dtype)
    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if disable_safety_checker:
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False

    pipe = AutoPipelineForImage2Image.from_pretrained(model_id, **kwargs).to(device)

    if enable_attention_slicing:
        pipe.enable_attention_slicing()

    # Cache for reuse
    _PIPELINE_CACHE[cache_key] = pipe
    logger.info("Cached SD pipeline for %s", cache_key)

    return pipe
