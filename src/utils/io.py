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


def save_csv(metrics: dict[str, float], path: Path) -> None:
    """Write a flat *metrics* mapping as a two-column CSV (*metric*, *value*).

    Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def snapshot_config(config: Any, path: Path) -> None:
    """Persist an OmegaConf config (or any supported object) as YAML.

    Parent directories are created automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)
