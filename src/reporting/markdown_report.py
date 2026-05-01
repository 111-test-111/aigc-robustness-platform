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

    # Load structured metrics if available
    structured_path = experiment_dir / "structured_metrics.json"
    structured_attacks: dict = {}
    structured_defenses: dict = {}
    if structured_path.exists():
        with open(structured_path) as f:
            structured = json.load(f)
            structured_attacks = structured.get("attacks", {})
            structured_defenses = structured.get("defenses", {})

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

    # Resolve seeds list for display
    seeds = _resolve_seeds_display(cfg.task)

    # Render template
    content = template.render(
        task_name=cfg.task.name,
        seeds=seeds,
        device=cfg.task.device,
        dataset_name=cfg.dataset.name,
        num_samples=cfg.dataset.num_samples,
        image_size=cfg.dataset.image_size,
        model_name=cfg.target_model.name,
        model_weights=cfg.target_model.weights,
        attacks=attacks,
        defenses=defenses,
        metrics=metrics,
        structured_attacks=structured_attacks,
        structured_defenses=structured_defenses,
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


def _resolve_seeds_display(task_cfg) -> list[int]:
    """Resolve seeds list from config for display in the report."""
    if "seeds" in task_cfg and task_cfg.seeds is not None:
        seeds = OmegaConf.to_container(task_cfg.seeds, resolve=True)
        if isinstance(seeds, list):
            return [int(s) for s in seeds]
    if "seed" in task_cfg and task_cfg.seed is not None:
        return [int(task_cfg.seed)]
    return [42]


def _fmt_mean_std(info: dict, key: str) -> str:
    """Format a metric value as mean ± std or plain value."""
    mean_key = f"{key}_mean"
    std_key = f"{key}_std"
    if mean_key in info:
        mean_val = info[mean_key]
        std_val = info.get(std_key, 0.0)
        return f"{mean_val:.4f} ± {std_val:.4f}"
    if key in info and isinstance(info[key], (int, float)):
        return f"{info[key]:.4f}"
    return "-"


def _generate_conclusions(metrics: dict, cfg) -> list[str]:
    """Auto-generate conclusion lines from metrics.

    Handles both raw metric keys (single-seed) and ``_mean`` suffixed
    variants (multi-seed aggregation).
    """
    conclusions: list[str] = []

    asr_keys = [k for k in metrics if k.endswith("_asr_on_clean_correct")
                or k.endswith("_asr_on_clean_correct_mean")]
    if not asr_keys:
        asr_keys = [
            k
            for k in metrics
            if (k.endswith("_asr") or k.endswith("_asr_mean"))
            and "_untargeted" not in k
            and "_targeted" not in k
        ]
    for key in asr_keys:
        if key.endswith("_asr_on_clean_correct_mean"):
            attack_name = key.removesuffix("_asr_on_clean_correct_mean")
        elif key.endswith("_asr_on_clean_correct"):
            attack_name = key.removesuffix("_asr_on_clean_correct")
        elif key.endswith("_asr_mean"):
            attack_name = key.removesuffix("_asr_mean")
        else:
            attack_name = key.removesuffix("_asr")
        asr = metrics[key]
        if asr > 0.8:
            conclusions.append(f"{attack_name} 攻击效果显著 (ASR={asr:.2%})")
        elif asr > 0.5:
            conclusions.append(f"{attack_name} 攻击效果中等 (ASR={asr:.2%})")
        else:
            conclusions.append(f"{attack_name} 攻击效果有限 (ASR={asr:.2%})")

    # Defense effectiveness
    ra_keys = [k for k in metrics if k.endswith("_robust_accuracy")
               or k.endswith("_robust_accuracy_mean")]
    for key in ra_keys:
        if key.endswith("_robust_accuracy_mean"):
            prefix = key.replace("_robust_accuracy_mean", "")
        else:
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
