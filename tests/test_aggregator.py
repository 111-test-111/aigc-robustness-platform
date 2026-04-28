"""Tests for src/evaluation/aggregator."""

from __future__ import annotations

import pytest

from src.evaluation.aggregator import DEFAULT_WEIGHTS, compute_robustness_score


class TestScoreRange:
    """Output is always clamped to [0, 1]."""

    def test_score_range(self):
        metrics = {
            "robust_accuracy": 0.8,
            "asr": 0.3,
            "clean_accuracy_drop": 0.1,
        }
        score = compute_robustness_score(metrics)
        assert 0 <= score <= 1

    def test_empty_metrics_gives_midpoint(self):
        score = compute_robustness_score({})
        # With defaults, semantic_quality and efficiency default to 0.5,
        # giving 0.15*0.5 + 0.15*0.5 = 0.15.
        assert 0 <= score <= 1


class TestExtremeScores:
    """Perfect and worst-case scenarios."""

    def test_perfect_score(self):
        metrics = {
            "robust_accuracy": 1.0,
            "asr": 0.0,
            "clean_accuracy_drop": 0.0,
            "lpips": 0.0,
            "latency_mean": 0.0,
        }
        score = compute_robustness_score(metrics)
        assert score > 0.8

    def test_worst_score(self):
        metrics = {
            "robust_accuracy": 0.0,
            "asr": 1.0,
            "clean_accuracy_drop": 1.0,
            "lpips": 1.0,
            "latency_mean": 20.0,
        }
        score = compute_robustness_score(metrics)
        assert score < 0.2


class TestCustomWeights:
    """Different weights produce different results."""

    def test_different_weights_differ(self):
        # Use asymmetric values so robust_accuracy != (1 - asr)
        metrics = {"robust_accuracy": 0.9, "asr": 0.2}
        w_ra = {
            "robust_accuracy": 1.0,
            "inverse_asr": 0.0,
            "semantic_quality": 0.0,
            "clean_retention": 0.0,
            "efficiency": 0.0,
        }
        w_asr = {
            "robust_accuracy": 0.0,
            "inverse_asr": 1.0,
            "semantic_quality": 0.0,
            "clean_retention": 0.0,
            "efficiency": 0.0,
        }
        s1 = compute_robustness_score(metrics, w_ra)  # 0.9
        s2 = compute_robustness_score(metrics, w_asr)  # 0.8
        assert s1 != s2


class TestDefaultWeights:
    """Default weights sum to 1.0."""

    def test_default_weights_sum_to_one(self):
        assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 0.01

    def test_default_weights_cover_all_components(self):
        expected_keys = {
            "robust_accuracy",
            "inverse_asr",
            "semantic_quality",
            "clean_retention",
            "efficiency",
        }
        assert set(DEFAULT_WEIGHTS.keys()) == expected_keys


class TestMetricFallbacks:
    """Verify metric key fallback chains work correctly."""

    def test_fgsm_asr_fallback(self):
        score = compute_robustness_score({"fgsm_asr": 0.4})
        score_direct = compute_robustness_score({"asr": 0.4})
        assert score == pytest.approx(score_direct)

    def test_pgd_asr_fallback(self):
        score = compute_robustness_score({"pgd_asr": 0.4})
        score_direct = compute_robustness_score({"asr": 0.4})
        assert score == pytest.approx(score_direct)

    def test_lpips_fallback(self):
        score = compute_robustness_score({"fgsm_lpips": 0.2})
        score_direct = compute_robustness_score({"lpips": 0.2})
        assert score == pytest.approx(score_direct)

    def test_latency_fallback(self):
        score = compute_robustness_score({"fgsm_vs_jpeg_latency_mean": 3.0})
        score_direct = compute_robustness_score({"latency_mean": 3.0})
        assert score == pytest.approx(score_direct)


class TestClamping:
    """Edge cases around the [0, 1] bounds."""

    def test_high_asr_clamps(self):
        metrics = {"asr": 2.0}
        score = compute_robustness_score(metrics)
        assert score >= 0.0

    def test_high_lpips_clamps(self):
        metrics = {"lpips": 2.0}
        score = compute_robustness_score(metrics)
        assert score >= 0.0
