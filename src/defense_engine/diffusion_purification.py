"""Diffusion purification defense (DiffPure, Nie et al., NeurIPS 2022).

Core idea:
  1. Add small Gaussian noise to the adversarial image (forward diffusion).
  2. Denoise to recover a clean version (reverse diffusion).
The noise destroys adversarial perturbations; the reverse process
reconstructs a clean image.

When a full DDPM model is unavailable, a Stable-Diffusion img2img pipeline
with low denoising strength serves as a practical substitute.  If that
pipeline is also unavailable, a simple iterative Gaussian-blur fallback is
used so the defense can still execute.
"""

from __future__ import annotations

import logging
import time

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import GaussianBlur

from src.defense_engine.base import Defense, DefenseResult

logger = logging.getLogger(__name__)


class DiffusionPurificationDefense(Defense):
    """Apply diffusion-based purification to adversarial images.

    Supports two backends:
    - ``sd``: Real Stable Diffusion img2img with low strength (primary)
    - ``mock``: Iterative Gaussian blur denoising (for testing/offline use)

    When SD is unavailable and backend is ``sd``, falls back to blur with
    a warning. The fallback is recorded in metadata.
    """

    name = "diffusion_purification"

    def apply(self, batch: torch.Tensor, config: dict) -> DefenseResult:
        """Apply diffusion purification to a batch of images.

        Args:
            batch: (B, C, H, W) input images in [0, 1]
            config: defence-specific configuration with keys:
                - noise_level (float): sigma_t for forward noise, default 0.1
                - steps (int): reverse-diffusion steps, default 50
                - model_id (str): diffusion model ID for SD pipeline
                - backend (str): "sd" or "mock", default "sd"

        Returns:
            DefenseResult with defended samples and wall-clock latency.
        """
        noise_level: float = config.get("noise_level", 0.1)
        steps: int = config.get("steps", 50)
        model_id: str = config.get(
            "model_id", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        )
        backend: str = config.get("backend", "sd")

        start = time.perf_counter()

        # Step 1: Forward diffusion -- add Gaussian noise
        noise = torch.randn_like(batch) * noise_level
        noisy = torch.clamp(batch + noise, 0, 1)

        # Step 2: Reverse diffusion -- denoise
        actual_backend = backend
        if backend == "mock":
            defended = self._denoise_simple(noisy, steps)
            actual_backend = "gaussian_blur_fallback"
        else:
            try:
                defended = self._denoise_with_sd(noisy, model_id, steps, batch.device)
                actual_backend = "stable_diffusion"
            except (ImportError, RuntimeError, OSError) as exc:
                logger.warning(
                    "SD pipeline unavailable (%s); falling back to iterative blur denoising",
                    exc,
                )
                defended = self._denoise_simple(noisy, steps)
                actual_backend = "gaussian_blur_fallback"

        if batch.device.type == "cuda":
            torch.cuda.synchronize()
        latency = time.perf_counter() - start

        return DefenseResult(
            defended=torch.clamp(defended, 0, 1),
            latency_sec=latency,
            metadata={
                "configured_backend": backend,
                "actual_backend": actual_backend,
                "model_id": model_id,
                "noise_level": noise_level,
                "steps": steps,
            },
        )

    # ------------------------------------------------------------------
    # Denoising backends
    # ------------------------------------------------------------------

    def _denoise_with_sd(
        self,
        noisy: torch.Tensor,
        model_id: str,
        steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Denoise using a Stable-Diffusion img2img pipeline with low strength."""
        from torchvision.transforms import ToPILImage

        from src.model_zoo.generators import load_sd_pipeline

        pipe = load_sd_pipeline(model_id, device)
        to_pil = ToPILImage()

        denoised: list[torch.Tensor] = []
        denoise_strength = max(0.3, min(0.9, 1.0 / max(steps, 1)))
        for img in noisy:
            pil_img = to_pil(img.clamp(0, 1).cpu())
            result = pipe(
                prompt="a clean photo",
                image=pil_img,
                strength=denoise_strength,
                num_inference_steps=steps,
            )
            # Convert PIL image to tensor correctly
            tensor = TF.to_tensor(result.images[0]).float()
            denoised.append(tensor)

        return torch.stack(denoised).to(device)

    def _denoise_simple(self, noisy: torch.Tensor, steps: int) -> torch.Tensor:
        """Iterative Gaussian-blur denoising as a lightweight fallback."""
        blur = GaussianBlur(kernel_size=3, sigma=0.5)
        result = noisy.clone()
        # Cap iterations for speed; real DDPM uses many more steps
        for _ in range(min(steps, 5)):
            result = blur(result)
        return result
