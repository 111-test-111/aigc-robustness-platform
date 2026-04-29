"""Bit-depth reduction defense: destroy adversarial perturbations via quantization."""

import time

import torch

from src.defense_engine.base import Defense, DefenseResult


class BitDepthDefense(Defense):
    """Reduce pixel bit depth to disrupt adversarial perturbations."""

    name = "bit_depth"

    def apply(self, batch: torch.Tensor, config: dict) -> DefenseResult:
        """Quantize images to the specified bit depth.

        Args:
            batch: (B, C, H, W) input images in [0, 1]
            config: accepts "bits" (int, default 4)

        Returns:
            DefenseResult with quantized images and wall-clock latency
        """
        bits: int = config.get("bits", 4)
        levels = 2 ** bits

        start = time.perf_counter()
        defended = torch.round(batch * (levels - 1)) / (levels - 1)
        defended = torch.clamp(defended, 0, 1)
        latency = time.perf_counter() - start

        return DefenseResult(
            defended=defended,
            latency_sec=latency,
            metadata={"bits": bits, "levels": levels},
        )
