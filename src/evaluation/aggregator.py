"""Comprehensive robustness score aggregation.

Combines multiple experiment metrics into a single [0, 1] score using
configurable weights. This is an auxiliary metric for report display
and does not replace raw per-metric reporting.
"""

from __future__ import annotations

DEFAULT_WEIGHTS: dict[str, float] = {
    "robust_accuracy": 0.35,
    "inverse_asr": 0.20,
    "semantic_quality": 0.15,
    "clean_retention": 0.15,
    "efficiency": 0.15,
}

# Latency cap in seconds; values above this are clamped to 0 efficiency.
_LATENCY_CAP = 10.0


def compute_robustness_score(
    metrics: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted robustness score from experiment metrics.

    Args:
        metrics: dict containing metric values (asr, robust_accuracy, etc.)
        weights: optional custom weights, defaults to DEFAULT_WEIGHTS

    Returns:
        Score in [0, 1] (higher = more robust).
    """
    w = weights or DEFAULT_WEIGHTS

    # --- Robust accuracy ---------------------------------------------------
    ra = metrics.get("robust_accuracy", _suffix_best(metrics, "_robust_accuracy", max, 0.0))

    # --- Inverse ASR (attack success rate) ---------------------------------
    asr = metrics.get(
        "asr",
        _suffix_best(
            metrics,
            "_asr_on_clean_correct",
            max,
            _suffix_best(metrics, "_untargeted_asr", max, _suffix_best(metrics, "_asr", max, 0.0)),
        ),
    )

    # --- Semantic quality --------------------------------------------------
    # LPIPS is a perceptual distance (lower = better).  Convert to a
    # quality score in [0, 1] where 1 = perfect preservation.
    lpips = metrics.get("lpips", _suffix_best(metrics, "_lpips", min, None))
    if lpips is not None:
        semantic_quality: float = max(0.0, 1.0 - lpips)
    else:
        semantic_quality = 0.5

    # --- Clean accuracy retention ------------------------------------------
    # clean_accuracy_drop measures how much accuracy dropped on clean data
    # after adversarial training.  A drop of 0 means perfect retention.
    cad = metrics.get(
        "clean_accuracy_drop",
        _suffix_best(metrics, "_clean_accuracy_drop", max, 0.0),
    )

    # --- Efficiency --------------------------------------------------------
    # Normalise latency: 0 s -> 1.0, >= LATENCY_CAP s -> 0.0.
    latency = metrics.get(
        "latency_mean",
        _suffix_best(metrics, "_latency_mean", min, None),
    )
    if latency is not None:
        efficiency: float = max(0.0, 1.0 - latency / _LATENCY_CAP)
    else:
        efficiency = 0.5

    # --- Weighted sum ------------------------------------------------------
    score = (
        w.get("robust_accuracy", 0) * ra
        + w.get("inverse_asr", 0) * (1 - asr)
        + w.get("semantic_quality", 0) * semantic_quality
        + w.get("clean_retention", 0) * max(0.0, 1.0 - cad)
        + w.get("efficiency", 0) * efficiency
    )

    return float(max(0.0, min(1.0, score)))


def _suffix_best(metrics: dict[str, float], suffix: str, reducer, default):
    """Return a reduced value for numeric metrics ending with *suffix*."""
    values = [
        float(v)
        for k, v in metrics.items()
        if k.endswith(suffix) and isinstance(v, (int, float))
    ]
    if not values:
        return default
    return reducer(values)
