"""Tests for third-party progress-bar controls."""

import os

from src.progress import (
    configure_third_party_progress,
    show_progress_bars,
    third_party_progress_enabled,
)


def test_progress_bars_are_hidden_by_default(monkeypatch) -> None:
    """Third-party progress bars are suppressed unless explicitly requested."""
    monkeypatch.delenv("AIGC_SHOW_PROGRESS", raising=False)
    assert not show_progress_bars()
    assert not third_party_progress_enabled()


def test_progress_bars_can_be_enabled(monkeypatch) -> None:
    """AIGC_SHOW_PROGRESS=1 restores third-party progress bars."""
    monkeypatch.setenv("AIGC_SHOW_PROGRESS", "1")
    assert show_progress_bars()
    assert third_party_progress_enabled()


def test_configure_progress_sets_common_disable_envs(monkeypatch) -> None:
    """Progress suppression should use common library environment switches."""
    monkeypatch.delenv("AIGC_SHOW_PROGRESS", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("DISABLE_PROGRESS_BAR", raising=False)
    monkeypatch.delenv("TQDM_DISABLE", raising=False)

    configure_third_party_progress()

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["DISABLE_PROGRESS_BAR"] == "1"
    assert os.environ["TQDM_DISABLE"] == "1"
