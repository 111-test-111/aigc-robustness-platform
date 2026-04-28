"""Generator model loaders for diffusion pipelines."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def load_sd_pipeline(
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    device: torch.device = torch.device("cpu"),
    enable_attention_slicing: bool = True,
    torch_dtype: torch.dtype = torch.float16,
) -> Any:
    """Load a Stable Diffusion img2img pipeline.

    Args:
        model_id: HuggingFace model identifier or local path.
        device: Target device for the pipeline.
        enable_attention_slicing: Reduce VRAM usage via attention slicing.
        torch_dtype: Precision for model weights.

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

    logger.info("Loading SD pipeline from %s (dtype=%s)", model_id, torch_dtype)
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    ).to(device)

    if enable_attention_slicing:
        pipe.enable_attention_slicing()

    return pipe
