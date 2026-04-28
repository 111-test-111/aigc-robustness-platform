"""Perceptual quality metrics for evaluating adversarial samples."""

from __future__ import annotations

import torch

_lpips_fn: torch.nn.Module | None = None


def _get_lpips(device: torch.device) -> torch.nn.Module:
    """Return a cached LPIPS model, initializing on first call.

    Args:
        device: Torch device to place the model on.

    Returns:
        LPIPS model instance.

    Raises:
        ImportError: If the ``lpips`` package is not installed.
    """
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips
        except ImportError:
            raise ImportError(
                "lpips is required. Install with: pip install lpips"
            )
        _lpips_fn = lpips.LPIPS(net="alex").to(device)
    return _lpips_fn


def compute_lpips(clean: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Compute LPIPS perceptual distance between clean and adversarial images.

    Uses AlexNet as the backbone (``lpips.LPIPS(net='alex')``).

    Args:
        clean: Batch of clean images ``(B, C, H, W)`` in ``[0, 1]``.
        adversarial: Batch of adversarial images ``(B, C, H, W)`` in ``[0, 1]``.

    Returns:
        Mean LPIPS distance across the batch. 0 means identical, higher means
        more perceptually different.
    """
    clean_scaled = clean * 2 - 1
    adv_scaled = adversarial * 2 - 1

    fn = _get_lpips(clean.device)
    with torch.no_grad():
        distances = fn(clean_scaled, adv_scaled)
    return float(distances.mean().item())


def compute_fid(real_images: torch.Tensor, generated_images: torch.Tensor) -> float:
    """Compute Frechet Inception Distance between real and generated images.

    Args:
        real_images: (B, C, H, W) images in [0, 1]
        generated_images: (B, C, H, W) images in [0, 1]

    Returns:
        FID score (lower = more similar distributions)
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        raise ImportError(
            "torchmetrics[image] required. Install: pip install 'torchmetrics[image]'"
        )

    # torchmetrics FID expects uint8 images [0, 255]
    real_uint8 = (real_images * 255).to(torch.uint8)
    gen_uint8 = (generated_images * 255).to(torch.uint8)

    fid = FrechetInceptionDistance(feature=64)  # smaller feature dim for speed
    fid.update(real_uint8, real=True)
    fid.update(gen_uint8, real=False)
    return float(fid.compute().item())


def compute_clip_score(images: torch.Tensor, prompt: str) -> float:
    """Compute CLIP score between images and text prompt.

    Args:
        images: (B, C, H, W) images in [0, 1]
        prompt: text description

    Returns:
        CLIP score normalized to [0, 1] (higher = more consistent)
    """
    try:
        import torch.nn.functional as F
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError(
            "transformers required. Install: pip install transformers"
        )

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Convert tensor images to PIL Images
    pil_images = []
    for i in range(images.shape[0]):
        img = images[i]  # (C, H, W)
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        pil_images.append(Image.fromarray(img_np))

    inputs = processor(
        text=[prompt], images=pil_images, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # Cosine similarity between image and text embeddings, scaled by 100
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, dim=-1)
        similarity = (image_embeds @ text_embeds.T).diag()  # per-image scores
        # CLIP score = 100 * cosine_similarity, then normalize to [0, 1]
        clip_scores = similarity * 100.0

    return float(clip_scores.mean().item() / 100.0)  # normalize from [0,100] to [0,1]
