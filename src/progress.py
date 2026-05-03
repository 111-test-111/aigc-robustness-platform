"""Progress-bar controls for noisy third-party model loading."""

from __future__ import annotations

import contextlib
import io
import os
from collections.abc import Iterator

SHOW_PROGRESS_ENV = "AIGC_SHOW_PROGRESS"


def show_progress_bars() -> bool:
    """Return True when third-party download/model-loading bars should be shown."""
    return os.environ.get(SHOW_PROGRESS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def third_party_progress_enabled() -> bool:
    """Alias for APIs that expect an explicit progress boolean."""
    return show_progress_bars()


def configure_third_party_progress() -> None:
    """Suppress common library progress bars unless explicitly requested.

    The platform still prints its own phase transitions and OK/FAIL messages.
    This only targets tqdm-style progress output from downloads and model
    loading in Hugging Face, diffusers, transformers, and torch helpers.
    """
    if show_progress_bars():
        return

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("DISABLE_PROGRESS_BAR", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass

    try:
        from diffusers.utils import logging as diffusers_logging

        diffusers_logging.disable_progress_bar()
    except Exception:
        pass


@contextlib.contextmanager
def suppress_third_party_output() -> Iterator[None]:
    """Hide noisy stdout/stderr from third-party setup when progress is disabled."""
    if show_progress_bars():
        yield
        return

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        yield

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
    except Exception:
        pass
