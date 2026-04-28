"""Markdown report generation from experiment artifacts."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def generate_report(experiment_dir: Path) -> Path:
    """Generate a Markdown report from experiment artifacts.

    Reads ``config.yaml`` and ``metrics.json`` from *experiment_dir*,
    renders the Jinja2 template, and writes ``report.md`` to the same
    directory.

    Args:
        experiment_dir: Directory containing ``config.yaml`` and
            ``metrics.json``.

    Returns:
        Path to the generated ``report.md``.
    """
    from src.evaluation.aggregator import DEFAULT_WEIGHTS, compute_robustness_score

    config_path = experiment_dir / "config.yaml"
    metrics_path = experiment_dir / "metrics.json"

    # Load configuration
    cfg = OmegaConf.load(config_path)

    # Load metrics
    with open(metrics_path) as f:
        metrics: dict[str, float | str] = json.load(f)

    # Resolve attacks / defenses to plain dicts for the template
    attacks = [dict(a) for a in cfg.attacks]
    defenses = [dict(d) for d in cfg.defenses]

    # Check which figures exist
    figures_dir = experiment_dir / "figures"
    figures = {
        "sample_grid": (figures_dir / "sample_grid.png").exists(),
        "radar": (figures_dir / "radar.png").exists(),
        "metric_bars": (figures_dir / "metric_bars.png").exists(),
    }

    # Compute robustness score
    robustness_score = compute_robustness_score(metrics)

    # Generate conclusions
    conclusions = _generate_conclusions(metrics, cfg)

    # Setup Jinja2 environment
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
    )
    template = env.get_template("report.md.j2")

    # Render template
    content = template.render(
        task_name=cfg.task.name,
        seed=cfg.task.seed,
        device=cfg.task.device,
        dataset_name=cfg.dataset.name,
        num_samples=cfg.dataset.num_samples,
        image_size=cfg.dataset.image_size,
        model_name=cfg.target_model.name,
        model_weights=cfg.target_model.weights,
        attacks=attacks,
        defenses=defenses,
        metrics=metrics,
        robustness_score=robustness_score,
        weights=DEFAULT_WEIGHTS,
        figures=figures,
        conclusions=conclusions,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    # Write report
    report_path = experiment_dir / "report.md"
    report_path.write_text(content)
    logger.info("Report written to %s", report_path)
    return report_path


def _generate_conclusions(metrics: dict, cfg) -> list[str]:
    """Auto-generate conclusion lines from metrics."""
    conclusions: list[str] = []

    # Attack effectiveness
    asr_keys = [k for k in metrics if k.endswith("_asr")]
    for key in asr_keys:
        attack_name = key.replace("_asr", "")
        asr = metrics[key]
        if asr > 0.8:
            conclusions.append(f"{attack_name} 攻击效果显著 (ASR={asr:.2%})")
        elif asr > 0.5:
            conclusions.append(f"{attack_name} 攻击效果中等 (ASR={asr:.2%})")
        else:
            conclusions.append(f"{attack_name} 攻击效果有限 (ASR={asr:.2%})")

    # Defense effectiveness
    ra_keys = [k for k in metrics if k.endswith("_robust_accuracy")]
    for key in ra_keys:
        prefix = key.replace("_robust_accuracy", "")
        ra = metrics[key]
        if ra > 0.8:
            conclusions.append(f"{prefix} 防御后准确率良好 (RA={ra:.2%})")
        elif ra > 0.5:
            conclusions.append(f"{prefix} 防御后准确率中等 (RA={ra:.2%})")
        else:
            conclusions.append(f"{prefix} 防御后准确率较低 (RA={ra:.2%})")

    if not conclusions:
        conclusions.append("实验完成，详见指标表格。")

    return conclusions
