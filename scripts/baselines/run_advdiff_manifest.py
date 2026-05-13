#!/usr/bin/env python3
"""Run AdvDiff against a platform manifest.

This script is intentionally a thin bridge around the third-party
``EricDai0/advdiff`` code.  It is launched by the platform's external attack
adapter, reads the temporary manifest produced by the runner, generates one
class-conditional AdvDiff image for each manifest row, and saves PNG files to
the requested output paths.

AdvDiff is a class-conditional unrestricted generation baseline, not an
img2img attack.  The input images in the manifest are therefore used for their
labels and for output bookkeeping, while generation starts from noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV with index,input_path,label,output_path.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated PNGs.")
    parser.add_argument("--metadata", required=True, help="Path to write metadata JSON.")
    parser.add_argument("--advdiff-root", default="third_party/advdiff", help="Path to EricDai0/advdiff checkout.")
    parser.add_argument(
        "--config",
        default="configs/latent-diffusion/cin256-v2.yaml",
        help="AdvDiff latent diffusion config, relative to advdiff root unless absolute.",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/ldm/cin256-v2/model.ckpt",
        help="AdvDiff ImageNet LDM checkpoint, relative to advdiff root unless absolute.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4, help="Internal AdvDiff generation batch size.")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=3.0, help="Unconditional guidance scale.")
    parser.add_argument("--K", type=int, default=2, help="Adversarial prior refinement rounds.")
    parser.add_argument("--s", type=float, default=1.0, help="Adversarial guidance step size.")
    parser.add_argument("--a", type=float, default=0.5, help="Adversarial prior step size.")
    parser.add_argument(
        "--early-stop",
        choices=["all", "any", "none"],
        default="all",
        help="Stop K-loop after all/any/no samples are successful under the AdvDiff victim.",
    )
    parser.add_argument("--save-npz", default="", help="Optional path to save generated images and labels as NPZ.")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest/config paths without loading models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    advdiff_root = _resolve_path(args.advdiff_root, repo_root)
    config_path = _resolve_path(args.config, advdiff_root)
    checkpoint_path = _resolve_path(args.checkpoint, advdiff_root)
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    metadata_path = Path(args.metadata).resolve()

    rows = _read_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _write_json(metadata_path, {
            "dry_run": True,
            "rows": len(rows),
            "unique_labels": sorted({int(row["label"]) for row in rows}),
            "advdiff_root": str(advdiff_root),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
        })
        return

    if not advdiff_root.exists():
        raise FileNotFoundError(f"AdvDiff root not found: {advdiff_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"AdvDiff config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"AdvDiff checkpoint not found: {checkpoint_path}. "
            "Download cin256-v2/model.ckpt as described in third_party/advdiff/README.md."
        )

    start = time.monotonic()
    _seed_everything(args.seed)
    device = _resolve_device(args.device)
    _prepare_advdiff_imports(advdiff_root)

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from torchvision.utils import save_image  # noqa: F401 - imported to match advdiff runtime deps

    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim_adv import DDIMSampler

    config = OmegaConf.load(str(config_path))
    model = _load_ldm_model(instantiate_from_config, config, checkpoint_path, device)
    victim = _load_resnet50_victim(device)
    sampler = ManifestDDIMSampler(
        model,
        vic_model=victim,
        early_stop=args.early_stop,
        guidance_fraction=0.2,
    )

    generated_images: list[Any] = []
    generated_labels: list[int] = []
    victim_preds: list[int] = []
    victim_success: list[bool] = []

    with torch.no_grad():
        with model.ema_scope():
            for chunk in _chunks(rows, args.batch_size):
                labels = torch.tensor([int(row["label"]) for row in chunk], device=device, dtype=torch.long)
                uc = model.get_learned_conditioning({
                    model.cond_stage_key: torch.full_like(labels, 1000),
                })
                conditioning = model.get_learned_conditioning({
                    model.cond_stage_key: labels,
                })
                samples_ddim, _ = sampler.sample(
                    S=args.ddim_steps,
                    conditioning=conditioning,
                    batch_size=len(chunk),
                    shape=[3, 64, 64],
                    verbose=False,
                    unconditional_guidance_scale=args.scale,
                    unconditional_conditioning=uc,
                    eta=args.ddim_eta,
                    label=labels,
                    K=args.K,
                    s=args.s,
                    a=args.a,
                )
                decoded = model.decode_first_stage(samples_ddim)
                decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)

                preds = getattr(sampler, "last_pred", torch.full_like(labels, -1))
                success = getattr(sampler, "last_success", torch.zeros_like(labels, dtype=torch.bool))
                victim_preds.extend(int(v) for v in preds.detach().cpu().tolist())
                victim_success.extend(bool(v) for v in success.detach().cpu().tolist())

                for tensor, row in zip(decoded.detach().cpu(), chunk):
                    output_path = Path(row["output_path"])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    image = _tensor_to_pil(tensor)
                    image.save(output_path)
                    generated_images.append(tensor.permute(1, 2, 0).numpy())
                    generated_labels.append(int(row["label"]))

    if args.save_npz:
        np.savez(
            _resolve_path(args.save_npz, repo_root),
            images=np.stack(generated_images, axis=0),
            labels=np.asarray(generated_labels, dtype=np.int64),
        )

    elapsed = time.monotonic() - start
    metadata = {
        "method": "AdvDiff",
        "source_repo": "https://github.com/EricDai0/advdiff",
        "source_commit": _git_commit(advdiff_root),
        "advdiff_root": str(advdiff_root),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "ddim_steps": args.ddim_steps,
        "ddim_eta": args.ddim_eta,
        "scale": args.scale,
        "K": args.K,
        "s": args.s,
        "a": args.a,
        "early_stop": args.early_stop,
        "num_samples": len(rows),
        "unique_labels": sorted({int(row["label"]) for row in rows}),
        "queries_per_sample": int(args.ddim_steps * args.K),
        "queries_are_generation_steps": True,
        "victim_model": "torchvision.resnet50",
        "victim_predictions": victim_preds,
        "victim_success": victim_success,
        "victim_success_rate": float(sum(victim_success) / len(victim_success)) if victim_success else 0.0,
        "attack_semantics": "class_conditional_unrestricted_generation",
        "elapsed_sec": round(elapsed, 4),
    }
    _write_json(metadata_path, metadata)


class ManifestDDIMSampler:
    """DDIM sampler variant that preserves one output per manifest row."""

    def __init__(self, model, schedule: str = "linear", vic_model=None, early_stop: str = "all", guidance_fraction: float = 0.2):
        from ldm.models.diffusion.ddim_adv import DDIMSampler

        self._base = DDIMSampler(model, schedule=schedule, vic_model=vic_model)
        self.model = self._base.model
        self.vic_model = vic_model
        self.early_stop = early_stop
        self.guidance_fraction = guidance_fraction
        self.last_pred = None
        self.last_success = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def sample(self, *args, **kwargs):
        return self._base_sample(*args, **kwargs)

    def _base_sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        label=None,
        K=10,
        s=2.0,
        a=1.0,
        **kwargs,
    ):
        self._base.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        samples, intermediates = self.ddim_sampling(
            conditioning,
            (batch_size, *shape),
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            label=label,
            K=K,
            s=s,
            a=a,
        )
        return samples, intermediates

    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        label=None,
        K=10,
        s=0.75,
        a=0.5,
    ):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm

        device = self.model.betas.device
        batch_size = shape[0]
        img = torch.randn(shape, device=device) if x_T is None else x_T

        if timesteps is None:
            timesteps = self.model.num_timesteps if ddim_use_original_steps else self._base.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self._base.ddim_timesteps.shape[0], 1) * self._base.ddim_timesteps.shape[0]) - 1
            timesteps = self._base.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        labels = label.to(device)
        prior_img = img.detach().requires_grad_(True)
        success = torch.zeros(batch_size, dtype=torch.bool, device=device)
        pred = torch.full_like(labels, -1)

        for round_idx in range(K):
            img = prior_img.detach().requires_grad_(True)
            iterator = tqdm(
                time_range,
                desc=f"AdvDiff DDIM round {round_idx + 1}/{K}",
                total=total_steps,
                leave=False,
            )
            last_prior_gradient = None
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((batch_size,), step, device=device, dtype=torch.long)
                if mask is not None:
                    if x0 is None:
                        raise ValueError("x0 is required when mask is provided.")
                    img_orig = self.model.q_sample(x0, ts)
                    img = img_orig * mask + (1.0 - mask) * img

                img, pred_x0 = self._base.p_sample_ddim(
                    img,
                    cond,
                    ts,
                    index=index,
                    use_original_steps=ddim_use_original_steps,
                    quantize_denoised=quantize_denoised,
                    temperature=temperature,
                    noise_dropout=noise_dropout,
                    score_corrector=score_corrector,
                    corrector_kwargs=corrector_kwargs,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                )

                if index > 0 and index <= total_steps * self.guidance_fraction:
                    with torch.enable_grad():
                        img_n = img.detach().requires_grad_(True)
                        decoded = self.model.differentiable_decode_first_stage(img_n)
                        decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
                        logits = self.vic_model(_resnet_input(decoded))
                        log_probs = F.log_softmax(logits, dim=-1)
                        target = _second_like_target(logits, labels)
                        selected = log_probs[torch.arange(len(logits), device=device), target]
                        gradient = torch.autograd.grad(selected.sum(), img_n)[0]
                    img = img + s * gradient.float()

                if callback:
                    callback(i)
                if img_callback:
                    img_callback(pred_x0, i)
                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates["x_inter"].append(img)
                    intermediates["pred_x0"].append(pred_x0)

            with torch.enable_grad():
                img_n = img.detach().requires_grad_(True)
                decoded_for_grad = self.model.differentiable_decode_first_stage(img_n)
                decoded_for_grad = torch.clamp((decoded_for_grad + 1.0) / 2.0, min=0.0, max=1.0)
                logits = self.vic_model(_resnet_input(decoded_for_grad))
                log_probs = F.log_softmax(logits, dim=-1)
                target = _second_like_target(logits, labels)
                selected = log_probs[torch.arange(len(logits), device=device), target]
                last_prior_gradient = torch.autograd.grad(selected.sum(), img_n)[0]

            decoded = self.model.decode_first_stage(img)
            decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
            logits = self.vic_model(_resnet_input(decoded))
            pred = torch.argmax(logits, dim=1)
            success = pred != labels
            if self.early_stop == "all" and bool(success.all()):
                break
            if self.early_stop == "any" and bool(success.any()):
                break
            if last_prior_gradient is not None:
                prior_img = prior_img + a * last_prior_gradient.float()

        self.last_pred = pred.detach()
        self.last_success = success.detach()
        return img, intermediates


def _prepare_advdiff_imports(advdiff_root: Path) -> None:
    _patch_torchvision_weight_enums()
    sys.path.insert(0, str(advdiff_root))
    sys.path.insert(0, str(advdiff_root / "taming-transformers"))


def _patch_torchvision_weight_enums() -> None:
    try:
        import torchvision.models as models
    except Exception:
        return

    if not hasattr(models, "ResNet50_Weights"):
        class _ResNet50Weights:
            DEFAULT = None
            IMAGENET1K_V1 = None
            IMAGENET1K_V2 = None

        models.ResNet50_Weights = _ResNet50Weights


def _load_ldm_model(instantiate_from_config, config, checkpoint_path: Path, device: str):
    import torch

    print(f"Loading AdvDiff LDM checkpoint from {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"AdvDiff LDM missing keys: {len(missing)}")
    if unexpected:
        print(f"AdvDiff LDM unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def _load_resnet50_victim(device: str):
    import torchvision.models as models

    if hasattr(models, "ResNet50_Weights") and getattr(models.ResNet50_Weights, "IMAGENET1K_V2", None) is not None:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        weights = "IMAGENET1K_V2"
    else:
        model = models.resnet50(pretrained=True)
        weights = "pretrained=True"
    print(f"Loaded AdvDiff victim ResNet50 ({weights})")
    return model.to(device).eval()


def _resnet_input(images):
    import torch
    import torch.nn.functional as F

    if images.shape[-2:] != (224, 224):
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def _second_like_target(logits, labels):
    import torch

    top2 = logits.argsort(dim=1, descending=True)[:, :2]
    return torch.where(labels == top2[:, 0], top2[:, 1], top2[:, 0])


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("AdvDiff requires CUDA, but torch.cuda.is_available() is False.")
        if ":" in requested:
            torch.cuda.set_device(int(requested.split(":", 1)[1]))
    if requested == "cpu":
        raise RuntimeError("AdvDiff generation is too expensive for CPU; use a CUDA device.")
    return requested


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"index", "input_path", "label", "output_path"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    return rows


def _chunks(rows: list[dict[str, str]], size: int):
    if size <= 0:
        raise ValueError("batch-size must be positive.")
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def _tensor_to_pil(tensor):
    import numpy as np
    from PIL import Image

    array = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def _git_commit(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
