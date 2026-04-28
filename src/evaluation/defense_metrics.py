"""Defense evaluation metrics."""

from __future__ import annotations

import statistics

import torch
import torch.nn as nn


def compute_robust_accuracy(
    model: nn.Module,
    defended: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute accuracy on defended (adversarial) samples.

    Args:
        model: Classification model.
        defended: Defended input tensor (batch, channels, height, width).
        labels: Ground-truth labels tensor (batch,).

    Returns:
        Accuracy on defended samples in [0, 1].
    """
    model.eval()
    with torch.no_grad():
        preds = model(defended).argmax(dim=1)
        return float((preds == labels).float().mean().item())


def compute_clean_accuracy_drop(
    model: nn.Module,
    clean: torch.Tensor,
    defended: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute drop in clean accuracy due to defense.

    A positive value means the defense reduced accuracy on clean samples.

    Args:
        model: Classification model.
        clean: Clean (unmodified) input tensor (batch, channels, height, width).
        defended: Defended input tensor (batch, channels, height, width).
        labels: Ground-truth labels tensor (batch,).

    Returns:
        Accuracy drop (clean_acc - defended_acc).
    """
    model.eval()
    with torch.no_grad():
        clean_acc = (model(clean).argmax(dim=1) == labels).float().mean().item()
        defended_acc = (model(defended).argmax(dim=1) == labels).float().mean().item()
        return float(clean_acc - defended_acc)


def compute_latency(latencies: list[float]) -> dict[str, float]:
    """Compute latency statistics.

    Args:
        latencies: List of per-sample latency values in seconds.

    Returns:
        Dictionary with mean, median, max, and total statistics.
    """
    if not latencies:
        return {
            "mean": 0.0,
            "median": 0.0,
            "max": 0.0,
            "total": 0.0,
        }
    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "max": max(latencies),
        "total": sum(latencies),
    }
