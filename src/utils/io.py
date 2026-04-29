import csv
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def save_json(data: Any, path: Path) -> None:
    """Serialize *data* as pretty-printed JSON to *path*.

    Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# Metric key -> paper-friendly display name
_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "asr": "ASR (%)",
    "robust_accuracy": "Robust Accuracy (%)",
    "clean_accuracy_drop": "Clean Acc Drop (%)",
    "lpips": "LPIPS",
    "fid": "FID",
    "clip_score": "CLIP Score",
    "latency_mean": "Latency (s)",
    "queries_mean": "Queries (mean)",
    "queries_median": "Queries (median)",
    "queries_max": "Queries (max)",
}

# Keys whose float values are ratios and should be displayed as percentages
_PCT_SUFFIXES = ("asr", "accuracy", "drop")


def save_csv(metrics: dict[str, float], path: Path) -> None:
    """Write a flat *metrics* mapping as a paper-friendly two-column CSV.

    - Metric names are replaced with human-readable display names.
    - Ratio metrics containing ``asr``, ``accuracy``, or ``drop`` are
      multiplied by 100 and shown with 2 decimal places.
    - All other floats are formatted to 4 decimal places.

    Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            display_name = _METRIC_DISPLAY_NAMES.get(key, key)
            if isinstance(value, float):
                if any(s in key for s in _PCT_SUFFIXES):
                    display_value = f"{value * 100:.2f}"
                else:
                    display_value = f"{value:.4f}"
            else:
                display_value = str(value)
            writer.writerow([display_name, display_value])


def save_structured_csv(structured_metrics: dict[str, Any], path: Path) -> None:
    """Write structured attack/defense metrics as a paper-friendly wide CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    attacks = structured_metrics.get("attacks", {})
    defenses = structured_metrics.get("defenses", {})

    fieldnames = [
        "attack",
        "defense",
        "asr",
        "asr_on_clean_correct",
        "prediction_change_rate",
        "clean_accuracy",
        "adversarial_accuracy",
        "robust_accuracy",
        "clean_accuracy_drop",
        "clean_defended_accuracy",
        "lpips",
        "fid",
        "clip_score",
        "latency_sec",
        "backend",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        wrote_row = False
        for defense_key, defense_info in defenses.items():
            attack_name, defense_name = _split_defense_key(defense_key)
            attack_info = attacks.get(attack_name, {})
            metadata = defense_info.get("metadata", {})
            writer.writerow({
                "attack": attack_name,
                "defense": defense_name,
                "asr": _fmt(attack_info.get("asr")),
                "asr_on_clean_correct": _fmt(attack_info.get("asr_on_clean_correct")),
                "prediction_change_rate": _fmt(attack_info.get("prediction_change_rate")),
                "clean_accuracy": _fmt(attack_info.get("clean_accuracy")),
                "adversarial_accuracy": _fmt(attack_info.get("adversarial_accuracy")),
                "robust_accuracy": _fmt(defense_info.get("robust_accuracy")),
                "clean_accuracy_drop": _fmt(defense_info.get("clean_accuracy_drop")),
                "clean_defended_accuracy": _fmt(defense_info.get("clean_defended_accuracy")),
                "lpips": _fmt(attack_info.get("lpips")),
                "fid": _fmt(attack_info.get("fid")),
                "clip_score": _fmt(attack_info.get("clip_score")),
                "latency_sec": _fmt(defense_info.get("latency", {}).get("mean")),
                "backend": metadata.get("actual_backend") or metadata.get("backend") or "",
            })
            wrote_row = True

        if not wrote_row:
            for attack_name, attack_info in attacks.items():
                writer.writerow({
                    "attack": attack_name,
                    "defense": "",
                    "asr": _fmt(attack_info.get("asr")),
                    "asr_on_clean_correct": _fmt(attack_info.get("asr_on_clean_correct")),
                    "prediction_change_rate": _fmt(attack_info.get("prediction_change_rate")),
                    "clean_accuracy": _fmt(attack_info.get("clean_accuracy")),
                    "adversarial_accuracy": _fmt(attack_info.get("adversarial_accuracy")),
                    "robust_accuracy": "",
                    "clean_accuracy_drop": "",
                    "clean_defended_accuracy": "",
                    "lpips": _fmt(attack_info.get("lpips")),
                    "fid": _fmt(attack_info.get("fid")),
                    "clip_score": _fmt(attack_info.get("clip_score")),
                    "latency_sec": "",
                    "backend": "",
                })


def _split_defense_key(key: str) -> tuple[str, str]:
    """Split a structured defense key of the form ``attack_vs_defense``."""
    if "_vs_" not in key:
        return key, ""
    attack, defense = key.split("_vs_", 1)
    return attack, defense


def _fmt(value: Any) -> str:
    """Format CSV numeric cells consistently."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def snapshot_config(config: Any, path: Path) -> None:
    """Persist an OmegaConf config (or any supported object) as YAML.

    Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)
