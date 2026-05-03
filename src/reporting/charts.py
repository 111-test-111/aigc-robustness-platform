"""Chart generation for sample comparison, metric bars, and radar plots."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
warnings.filterwarnings("ignore", message=".*Glyph .* missing from font.*")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Configure matplotlib once
_FONT_CONFIGURED = False

# 宋体 fallback chain across platforms
_SONGTI_CANDIDATES = [
    "SimSong",
    "Songti SC",
    "STSong",
    "SimSun",
    "Songti TC",
    "STFangsong",
    "Noto Serif CJK SC",
    "WenQuanYi Zen Hei",
]


def _configure_plot_font() -> None:
    """Configure matplotlib for Chinese (宋体) chart rendering."""
    global _FONT_CONFIGURED
    if _FONT_CONFIGURED:
        return

    font_path = None
    for name in _SONGTI_CANDIDATES:
        for f in fm.fontManager.ttflist:
            if f.name == name and any(
                tag in f.fname.lower() for tag in ["song", "simsun", "st", "cjk"]
            ):
                font_path = f.fname
                break
        if font_path:
            break

    if font_path:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
        logger.debug("Using Chinese font: %s (%s)", prop.get_name(), font_path)
    else:
        plt.rcParams["font.sans-serif"] = _SONGTI_CANDIDATES + list(
            plt.rcParams.get("font.sans-serif", [])
        )
        logger.debug("No 宋体 found; using sans-serif fallback chain")

    plt.rcParams["axes.unicode_minus"] = False
    _FONT_CONFIGURED = True


def generate_sample_grid(
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    defended: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    metrics: dict | None = None,
    save_path: Path | str = "sample_grid.png",
    max_samples: int = 8,
    title: str = "样本对比",
) -> Path:
    """Generate a grid comparing clean, adversarial, and optionally defended samples.

    Args:
        clean: (B, C, H, W) original images in [0, 1]
        adversarial: (B, C, H, W) adversarial images in [0, 1]
        defended: (B, C, H, W) defended images in [0, 1], optional
        labels: (B,) labels, optional
        metrics: dict of metric name -> value, optional
        save_path: output file path
        max_samples: maximum samples to show
        title: figure title

    Returns:
        Path to saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_plot_font()

    n = min(clean.shape[0], max_samples)
    has_defended = defended is not None
    ncols = 3 if has_defended else 2

    fig, axes = plt.subplots(n, ncols, figsize=(4 * ncols, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["原始图像", "对抗样本"] + (["防御后"] if has_defended else [])
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=14, fontweight="bold")

    for i in range(n):
        # Clean
        img = clean[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")

        # Adversarial
        img = adversarial[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        axes[i, 1].imshow(img)
        axes[i, 1].axis("off")

        # Defended
        if has_defended:
            img = defended[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
            axes[i, 2].imshow(img)
            axes[i, 2].axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def generate_metric_bars(
    metrics_list: list[dict],
    labels: list[str],
    save_path: Path | str = "metric_bars.png",
    title: str = "指标对比",
    stds_list: list[dict] | None = None,
) -> Path:
    """Generate grouped bar chart comparing metrics across methods.

    Args:
        metrics_list: list of dicts, each mapping metric name -> value.
        labels: label for each method.
        save_path: output file path.
        title: figure title.
        stds_list: optional list of dicts with standard deviations for error bars.

    Returns:
        Path to saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_plot_font()

    if not metrics_list or not metrics_list[0]:
        return save_path

    metric_names = list(metrics_list[0].keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(metrics_list)

    fig, ax = plt.subplots(figsize=(max(8, len(metric_names) * 1.5), 6))

    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        values = [metrics.get(m, 0) for m in metric_names]
        stds = None
        if stds_list and i < len(stds_list) and stds_list[i]:
            stds = [stds_list[i].get(m, 0) for m in metric_names]
        ax.bar(x + i * width, values, width, label=label,
               yerr=stds, capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(x + width * (len(metrics_list) - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.set_ylabel("数值")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def generate_radar(
    metrics: dict[str, float],
    save_path: Path | str = "radar.png",
    title: str = "鲁棒性雷达图",
) -> Path:
    """Generate radar chart for multi-dimensional metrics.

    Args:
        metrics: mapping of metric name -> normalised value in [0, 1].
        save_path: output file path.
        title: figure title.

    Returns:
        Path to saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_plot_font()

    categories = list(metrics.keys())
    values = list(metrics.values())
    n = len(categories)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values_closed, alpha=0.25)
    ax.plot(angles, values_closed, "o-", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path
