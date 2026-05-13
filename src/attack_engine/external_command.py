"""External image-attack command adapters.

These adapters let the platform evaluate strong baselines implemented outside
this repository without vendoring fragile research code.  The platform owns the
experiment protocol: export the current batch, invoke an external command, read
back adversarial images, and compute all metrics through the existing runner.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.attack_engine.base import Attack, AttackResult


class ExternalImageCommandAttack(Attack):
    """Invoke an external image attack and load generated adversarial images.

    Expected command protocol
    -------------------------
    The adapter creates a temporary work directory containing:

    - ``inputs/*.png``: clean input images
    - ``manifest.csv``: ``index,input_path,label,output_path`` rows
    - ``labels.json``: label list
    - ``config.json``: attack config with command fields removed

    The command should write adversarial images either to the ``output_path``
    listed in ``manifest.csv`` or to a result manifest configured via
    ``result_manifest``.  The command may also write ``metadata.json`` with
    optional fields such as ``queries`` or ``queries_per_sample``.
    """

    name: str = "external"
    method_name: str = "external"

    def generate(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        target_model: torch.nn.Module,
        config: dict,
    ) -> AttackResult:
        command = config.get("command")
        if not command:
            raise ValueError(
                f"{self.name} requires a `command` config value. "
                "Use placeholders such as {manifest}, {input_dir}, "
                "{output_dir}, {config_json}, and {device}."
            )

        timeout_sec = config.get("timeout_sec")
        keep_work_dir = bool(config.get("keep_work_dir", False))
        resize_outputs = bool(config.get("resize_outputs", True))
        target_class: int | None = config.get("target_class")

        start_time = time.monotonic()
        work_dir = Path(tempfile.mkdtemp(prefix=f"{self.name}_"))

        try:
            input_dir = work_dir / "inputs"
            output_dir = work_dir / "outputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            manifest_path = work_dir / "manifest.csv"
            labels_path = work_dir / "labels.json"
            config_path = work_dir / "config.json"
            metadata_path = Path(config.get("metadata_path", output_dir / "metadata.json"))
            result_manifest_path = Path(config.get(
                "result_manifest",
                output_dir / "manifest.csv",
            ))

            expected_outputs = self._export_inputs(
                batch=batch,
                labels=labels,
                input_dir=input_dir,
                output_dir=output_dir,
                manifest_path=manifest_path,
            )
            labels_path.write_text(json.dumps(labels.detach().cpu().tolist(), indent=2))
            config_path.write_text(json.dumps(
                self._public_config(config),
                indent=2,
                default=str,
            ))

            argv = self._format_command(
                command=command,
                work_dir=work_dir,
                input_dir=input_dir,
                output_dir=output_dir,
                manifest_path=manifest_path,
                labels_path=labels_path,
                config_path=config_path,
                metadata_path=metadata_path,
                device=batch.device,
            )
            env = self._build_env(config, work_dir, input_dir, output_dir, manifest_path, batch.device)
            cwd = config.get("cwd")

            proc = subprocess.run(
                argv,
                cwd=str(cwd) if cwd else None,
                env=env,
                capture_output=True,
                text=True,
                timeout=float(timeout_sec) if timeout_sec is not None else None,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{self.name} external command failed with exit code "
                    f"{proc.returncode}.\nSTDOUT:\n{proc.stdout[-4000:]}\n"
                    f"STDERR:\n{proc.stderr[-4000:]}"
                )

            output_paths = self._resolve_output_paths(
                result_manifest_path=result_manifest_path,
                expected_outputs=expected_outputs,
            )
            adv = self._load_outputs(
                output_paths=output_paths,
                reference=batch,
                resize_outputs=resize_outputs,
            )
            external_metadata = self._read_metadata(metadata_path)
        finally:
            if not keep_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)

        if batch.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - start_time

        with torch.no_grad():
            orig_pred = target_model(batch).argmax(dim=1)
            adv_pred = target_model(adv).argmax(dim=1)
            if target_class is None:
                success = adv_pred != orig_pred
            else:
                success = adv_pred == int(target_class)

        queries = self._resolve_queries(
            external_metadata=external_metadata,
            config=config,
            batch_size=batch.shape[0],
        )

        metadata = {
            "backend": "external_command",
            "method": config.get("method", self.method_name),
            "command": self._redact_command(command),
            "cwd": config.get("cwd"),
            "target_class": target_class,
            "resize_outputs": resize_outputs,
            "work_dir": str(work_dir) if keep_work_dir else None,
            "external_metadata": external_metadata,
            "elapsed_sec": round(elapsed, 4),
        }

        return AttackResult(
            adversarial=adv,
            success=success,
            queries=queries,
            metadata=metadata,
        )

    @staticmethod
    def _export_inputs(
        batch: torch.Tensor,
        labels: torch.Tensor,
        input_dir: Path,
        output_dir: Path,
        manifest_path: Path,
    ) -> list[Path]:
        expected_outputs: list[Path] = []
        rows: list[dict[str, Any]] = []
        for idx, image in enumerate(batch.detach().cpu()):
            filename = f"{idx:06d}.png"
            input_path = input_dir / filename
            output_path = output_dir / filename
            TF.to_pil_image(image.clamp(0, 1)).save(input_path)
            expected_outputs.append(output_path)
            rows.append({
                "index": idx,
                "input_path": str(input_path),
                "label": int(labels[idx].detach().cpu().item()),
                "output_path": str(output_path),
            })

        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["index", "input_path", "label", "output_path"],
            )
            writer.writeheader()
            writer.writerows(rows)

        return expected_outputs

    @staticmethod
    def _public_config(config: dict) -> dict:
        hidden = {
            "command",
            "env",
            "cwd",
            "timeout_sec",
            "keep_work_dir",
            "metadata_path",
            "result_manifest",
        }
        return {k: v for k, v in config.items() if k not in hidden}

    @staticmethod
    def _format_command(
        command: str | list[str],
        work_dir: Path,
        input_dir: Path,
        output_dir: Path,
        manifest_path: Path,
        labels_path: Path,
        config_path: Path,
        metadata_path: Path,
        device: torch.device,
    ) -> list[str]:
        values = {
            "work_dir": str(work_dir),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "manifest": str(manifest_path),
            "labels_json": str(labels_path),
            "config_json": str(config_path),
            "metadata_json": str(metadata_path),
            "device": str(device),
        }
        parts = shlex.split(command) if isinstance(command, str) else list(command)
        return [str(part).format(**values) for part in parts]

    @staticmethod
    def _build_env(
        config: dict,
        work_dir: Path,
        input_dir: Path,
        output_dir: Path,
        manifest_path: Path,
        device: torch.device,
    ) -> dict[str, str]:
        env = os.environ.copy()
        extra_env = config.get("env", {})
        if isinstance(extra_env, dict):
            env.update({str(k): str(v) for k, v in extra_env.items()})
        env.update({
            "AIGC_ATTACK_WORK_DIR": str(work_dir),
            "AIGC_ATTACK_INPUT_DIR": str(input_dir),
            "AIGC_ATTACK_OUTPUT_DIR": str(output_dir),
            "AIGC_ATTACK_MANIFEST": str(manifest_path),
            "AIGC_ATTACK_DEVICE": str(device),
        })
        return env

    @staticmethod
    def _resolve_output_paths(
        result_manifest_path: Path,
        expected_outputs: list[Path],
    ) -> list[Path]:
        if not result_manifest_path.exists():
            return expected_outputs

        paths_by_index: dict[int, Path] = {}
        with open(result_manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                index = int(row.get("index", len(paths_by_index)))
                value = row.get("output_path") or row.get("path") or row.get("image_path")
                if value:
                    paths_by_index[index] = Path(value)

        return [
            paths_by_index.get(idx, expected_path)
            for idx, expected_path in enumerate(expected_outputs)
        ]

    @staticmethod
    def _load_outputs(
        output_paths: list[Path],
        reference: torch.Tensor,
        resize_outputs: bool,
    ) -> torch.Tensor:
        images: list[torch.Tensor] = []
        ref_h, ref_w = reference.shape[-2:]

        for output_path in output_paths:
            if not output_path.exists():
                raise FileNotFoundError(f"External attack did not produce {output_path}")
            with Image.open(output_path) as img:
                img = img.convert("RGB")
                if resize_outputs and img.size != (ref_w, ref_h):
                    img = img.resize((ref_w, ref_h), Image.Resampling.BICUBIC)
                elif img.size != (ref_w, ref_h):
                    raise ValueError(
                        f"External output {output_path} has size {img.size}, "
                        f"expected {(ref_w, ref_h)}. Set resize_outputs=true "
                        "to resize automatically."
                    )
                images.append(TF.to_tensor(img).float())

        return torch.stack(images).to(reference.device)

    @staticmethod
    def _read_metadata(metadata_path: Path) -> dict:
        if not metadata_path.exists():
            return {}
        try:
            data = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid external metadata JSON: {metadata_path}") from exc
        return data if isinstance(data, dict) else {"value": data}

    @staticmethod
    def _resolve_queries(
        external_metadata: dict,
        config: dict,
        batch_size: int,
    ) -> list[int]:
        queries = external_metadata.get("queries")
        if isinstance(queries, list) and len(queries) == batch_size:
            return [int(q) for q in queries]

        per_sample = external_metadata.get(
            "queries_per_sample",
            config.get("queries_per_sample", 0),
        )
        return [int(per_sample)] * batch_size

    @staticmethod
    def _redact_command(command: str | list[str]) -> str | list[str]:
        # Store the command shape for reproducibility without trying to parse
        # or redact arbitrary secrets from researcher-provided arguments.
        return command


class AdvDiffuserExternalAttack(ExternalImageCommandAttack):
    """External AdvDiffuser-compatible command adapter."""

    name: str = "advdiffuser"
    method_name: str = "AdvDiffuser"


class AdvDiffExternalAttack(ExternalImageCommandAttack):
    """External AdvDiff-compatible command adapter."""

    name: str = "advdiff"
    method_name: str = "AdvDiff"
