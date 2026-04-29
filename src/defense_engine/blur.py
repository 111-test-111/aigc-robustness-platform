"""Gaussian blur defense: destroy adversarial perturbations via spatial smoothing."""

import time

import torch
from torchvision.transforms import GaussianBlur as TVGaussianBlur

from src.defense_engine.base import Defense, DefenseResult


class GaussianBlurDefense(Defense):
    """Apply Gaussian blur to images to disrupt adversarial perturbations."""

    name = "gaussian_blur"

    def apply(self, batch: torch.Tensor, config: dict) -> DefenseResult:
        """Apply Gaussian blur to each image in the batch.

        Args:
            batch: (B, C, H, W) input images in [0, 1]
            config: accepts "kernel_size" (int, default 5) and "sigma" (float, default 1.0)

        Returns:
            DefenseResult with blurred images and wall-clock latency
        """
        kernel_size: int = config.get("kernel_size", 5)
        sigma: float = config.get("sigma", 1.0)

        start = time.perf_counter()
        blur = TVGaussianBlur(kernel_size=kernel_size, sigma=sigma)
        defended = blur(batch)
        latency = time.perf_counter() - start

        return DefenseResult(
            defended=defended,
            latency_sec=latency,
            metadata={"kernel_size": kernel_size, "sigma": sigma},
        )
