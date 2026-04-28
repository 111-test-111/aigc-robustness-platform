"""JPEG compression defense: destroy adversarial perturbations via lossy encoding."""

import io
import time

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.defense_engine.base import Defense, DefenseResult


class JPEGDefense(Defense):
    """Apply JPEG compression to images to disrupt adversarial perturbations."""

    name = "jpeg"

    def apply(self, batch: torch.Tensor, config: dict) -> DefenseResult:
        """Compress each image in the batch through JPEG encoding/decoding.

        Args:
            batch: (B, C, H, W) input images in [0, 1]
            config: must accept "quality" (int, 1-100, default 75)

        Returns:
            DefenseResult with JPEG-compressed images and wall-clock latency
        """
        quality: int = config.get("quality", 75)

        start = time.perf_counter()
        defended: list[torch.Tensor] = []
        for i in range(batch.shape[0]):
            img = TF.to_pil_image(batch[i].clamp(0, 1))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            defended_img = Image.open(buf).convert("RGB")
            defended.append(TF.to_tensor(defended_img))

        latency = time.perf_counter() - start
        return DefenseResult(
            defended=torch.stack(defended),
            latency_sec=latency,
        )
