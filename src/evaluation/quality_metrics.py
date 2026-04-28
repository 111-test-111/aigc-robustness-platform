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
