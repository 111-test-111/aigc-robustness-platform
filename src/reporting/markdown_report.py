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
    config_path = experiment_dir / "config.yaml"
    metrics_path = experiment_dir / "metrics.json"

    # Load configuration
    cfg = OmegaConf.load(config_path)

    # Load metrics
    with open(metrics_path) as f:
        metrics: dict[str, float | str] = json.load(f)

    # Extract attack ASRs for conclusion section
    attack_asrs: dict[str, float] = {
        k.replace("_asr", ""): v
        for k, v in metrics.items()
        if k.endswith("_asr")
    }

    # Resolve attacks / defenses to plain dicts for the template
    attacks = [dict(a) for a in cfg.attacks]
    defenses = [dict(d) for d in cfg.defenses]

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
        model_name=cfg.target_model.name,
        attacks=attacks,
        defenses=defenses,
        metrics=metrics,
        attack_asrs=attack_asrs,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    # Write report
    report_path = experiment_dir / "report.md"
    report_path.write_text(content)
    logger.info("Report written to %s", report_path)
    return report_path
