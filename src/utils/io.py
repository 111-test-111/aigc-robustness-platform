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


def snapshot_config(config: Any, path: Path) -> None:
    """Persist an OmegaConf config (or any supported object) as YAML.

    Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)
