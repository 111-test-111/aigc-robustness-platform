"""Attack evaluation metrics."""

from __future__ import annotations

import statistics

import torch


def compute_asr(success_mask: torch.Tensor) -> float:
    """Compute Attack Success Rate.

    Args:
        success_mask: Boolean tensor where True indicates a successful attack.

    Returns:
        Attack success rate in [0, 1].
    """
    if success_mask.numel() == 0:
        return 0.0
    return float(success_mask.float().mean().item())


def compute_queries(queries_list: list[int]) -> dict[str, float]:
    """Compute query count statistics.

    Args:
        queries_list: List of per-sample query counts.

    Returns:
        Dictionary with mean, median, max, and total statistics.
    """
    if not queries_list:
        return {
            "mean": 0.0,
            "median": 0.0,
            "max": 0,
            "total": 0,
        }
    return {
        "mean": statistics.mean(queries_list),
        "median": statistics.median(queries_list),
        "max": max(queries_list),
        "total": sum(queries_list),
    }
