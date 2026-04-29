import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import src.utils.device as device_utils
from src.utils.device import get_device
from src.utils.io import save_csv, save_json, snapshot_config
from src.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------

class TestSeedEverything:
    def test_reproducibility_python_random(self, tmp_path: Path) -> None:
        seed_everything(42)
        a = [random.random() for _ in range(5)]
        seed_everything(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_reproducibility_numpy(self) -> None:
        seed_everything(123)
        a = np.random.rand(5)
        seed_everything(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_reproducibility_torch_cpu(self) -> None:
        seed_everything(7)
        a = torch.randn(5)
        seed_everything(7)
        b = torch.randn(5)
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_explicit_cpu(self) -> None:
        dev = get_device("cpu")
        assert dev == torch.device("cpu")

    def test_mps_detection_requires_apple_silicon(self, monkeypatch) -> None:
        monkeypatch.setattr(device_utils.platform, "system", lambda: "Linux")
        monkeypatch.setattr(device_utils.platform, "machine", lambda: "x86_64")

        assert device_utils._is_supported_apple_silicon_mps() is False

    def test_auto_prefers_mps_on_supported_apple_silicon(self, monkeypatch) -> None:
        monkeypatch.setattr(
            device_utils, "_is_supported_apple_silicon_mps", lambda: True
        )

        dev = get_device("auto")
        assert dev == torch.device("mps")

    def test_auto_uses_cuda_on_non_apple_gpu_machine(self, monkeypatch) -> None:
        monkeypatch.setattr(
            device_utils, "_is_supported_apple_silicon_mps", lambda: False
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        dev = get_device("auto")
        assert dev == torch.device("cuda")

    def test_auto_falls_back_to_cpu_without_gpu(self, monkeypatch) -> None:
        monkeypatch.setattr(
            device_utils, "_is_supported_apple_silicon_mps", lambda: False
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        dev = get_device("auto")
        assert dev == torch.device("cpu")


# ---------------------------------------------------------------------------
# save_json
# ---------------------------------------------------------------------------

class TestSaveJson:
    def test_saves_and_round_trips(self, tmp_path: Path) -> None:
        data = {"name": "test", "values": [1, 2, 3]}
        out = tmp_path / "sub" / "data.json"
        save_json(data, out)
        loaded = json.loads(out.read_text())
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "c.json"
        save_json({"k": "v"}, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# save_csv
# ---------------------------------------------------------------------------

class TestSaveCsv:
    def test_saves_metrics(self, tmp_path: Path) -> None:
        metrics = {"accuracy": 0.95, "loss": 0.05}
        out = tmp_path / "metrics.csv"
        save_csv(metrics, out)
        with open(out, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["Metric", "Value"]

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "m.csv"
        save_csv({"x": 1.0}, out)
        assert out.exists()

    def test_paper_friendly_format(self, tmp_path: Path) -> None:
        metrics = {"asr": 0.8543, "lpips": 0.1234, "robust_accuracy": 0.72}
        path = tmp_path / "metrics.csv"
        save_csv(metrics, path)
        content = path.read_text()
        assert "ASR (%)" in content
        assert "85.43" in content
        assert "0.1234" in content
        assert "Robust Accuracy (%)" in content
        assert "72.00" in content


# ---------------------------------------------------------------------------
# snapshot_config
# ---------------------------------------------------------------------------

class TestSnapshotConfig:
    def test_saves_and_loads(self, tmp_path: Path) -> None:
        cfg = OmegaConf.create({"model": {"name": "resnet", "layers": 18}})
        out = tmp_path / "config.yaml"
        snapshot_config(cfg, out)
        loaded = OmegaConf.load(out)
        assert loaded.model.name == "resnet"
        assert loaded.model.layers == 18

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "cfg.yaml"
        snapshot_config(OmegaConf.create({"k": 1}), out)
        assert out.exists()
