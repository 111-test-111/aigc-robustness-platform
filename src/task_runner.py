"""Experiment orchestration pipeline.

Loads a YAML configuration, runs attacks and defenses against a target
model, computes evaluation metrics, and saves all artifacts to an output
directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.attack_engine.adv_gan import AdvGANAttack
from src.attack_engine.base import Attack, AttackResult
from src.attack_engine.diffusion_attack import DiffusionAttack
from src.attack_engine.fgsm import FGSMAttack
from src.attack_engine.pgd import PGDAttack
from src.defense_engine.base import Defense, DefenseResult
from src.defense_engine.bit_depth import BitDepthDefense
from src.defense_engine.blur import GaussianBlurDefense
from src.defense_engine.diffusion_purification import DiffusionPurificationDefense
from src.defense_engine.jpeg import JPEGDefense
from src.evaluation.attack_metrics import compute_asr, compute_queries
from src.evaluation.defense_metrics import (
    compute_clean_accuracy_drop,
    compute_latency,
    compute_robust_accuracy,
)
from src.evaluation.quality_metrics import compute_clip_score, compute_lpips
from src.model_zoo.classifiers import load_classifier
from src.reporting.charts import generate_sample_grid
from src.utils.device import get_device
from src.utils.io import save_csv, save_json, snapshot_config
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

ATTACK_REGISTRY: dict[str, type[Attack]] = {
    "fgsm": FGSMAttack,
    "pgd": PGDAttack,
    "advgan": AdvGANAttack,
    "diffusion": DiffusionAttack,
}

DEFENSE_REGISTRY: dict[str, type[Defense]] = {
    "jpeg": JPEGDefense,
    "gaussian_blur": GaussianBlurDefense,
    "bit_depth": BitDepthDefense,
    "diffusion_purification": DiffusionPurificationDefense,
}


def _load_dataset(cfg: DictConfig) -> Subset:
    """Load a dataset based on configuration.

    Currently supports ``cifar10`` via torchvision (auto-downloads if
    missing).

    Args:
        cfg: Dataset section of the experiment config.  Must contain
            ``name``, ``root``, ``num_samples``, and ``image_size``.

    Returns:
        A ``torch.utils.data.Subset`` with at most ``num_samples`` items.

    Raises:
        ValueError: If the dataset name is not recognised.
    """
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    if cfg.name == "cifar10":
        dataset = datasets.CIFAR10(
            root=cfg.root,
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")

    indices = list(range(min(cfg.num_samples, len(dataset))))
    return Subset(dataset, indices)


def _collect_tensor(loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Iterate over *loader* and return concatenated images and labels on *device*."""
    all_images: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for images, labels in loader:
        all_images.append(images)
        all_labels.append(labels)
    return torch.cat(all_images).to(device), torch.cat(all_labels).to(device)


def run_experiment(config_path: Path) -> Path:
    """Run a complete experiment from a YAML configuration file.

    Pipeline:
        1. Load and snapshot configuration.
        2. Set random seeds and resolve compute device.
        3. Load dataset and target model.
        4. For each attack: generate adversarial samples, compute attack
           metrics, then for each defense: apply defense, compute defense
           metrics.
        5. Persist all metrics (JSON + CSV) and return the output path.

    Args:
        config_path: Path to a YAML experiment configuration file.

    Returns:
        The ``Path`` of the output directory containing all artifacts.
    """
    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    cfg: DictConfig = OmegaConf.load(config_path)
    output_dir = Path(cfg.report.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    # Snapshot the resolved config for reproducibility
    snapshot_config(cfg, output_dir / "config.yaml")
    logger.info("Loaded config from %s -> output %s", config_path, output_dir)

    # ------------------------------------------------------------------
    # 2. Setup seed and device
    # ------------------------------------------------------------------
    seed_everything(cfg.task.seed)
    device = get_device(cfg.task.device)
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 3. Load data
    # ------------------------------------------------------------------
    dataset = _load_dataset(cfg.dataset)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    clean, labels = _collect_tensor(loader, device)
    logger.info("Loaded %d samples", clean.shape[0])

    # ------------------------------------------------------------------
    # 4. Load target model
    # ------------------------------------------------------------------
    model = load_classifier(
        cfg.target_model.name,
        cfg.target_model.weights,
        device,
    )
    logger.info("Loaded model: %s (weights=%s)", cfg.target_model.name, cfg.target_model.weights)

    # ------------------------------------------------------------------
    # 5. Build attack / defense instances
    # ------------------------------------------------------------------
    attack_methods: list[tuple[Attack, dict]] = []
    for a in cfg.attacks:
        attack_cls = ATTACK_REGISTRY.get(a.name)
        if attack_cls is None:
            raise ValueError(f"Unknown attack: {a.name}. Available: {list(ATTACK_REGISTRY)}")
        attack_methods.append((attack_cls(), OmegaConf.to_container(a, resolve=True)))  # type: ignore[arg-type]

    defense_methods: list[tuple[Defense, dict]] = []
    for d in cfg.defenses:
        defense_cls = DEFENSE_REGISTRY.get(d.name)
        if defense_cls is None:
            raise ValueError(f"Unknown defense: {d.name}. Available: {list(DEFENSE_REGISTRY)}")
        defense_methods.append((defense_cls(), OmegaConf.to_container(d, resolve=True)))  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # 6. Run attacks and defenses, collect metrics
    # ------------------------------------------------------------------
    all_metrics: dict[str, float] = {}

    for attack, attack_cfg in attack_methods:
        logger.info("Running attack: %s", attack.name)
        result: AttackResult = attack.generate(clean, labels, model, attack_cfg)

        # Attack-level metrics
        asr = compute_asr(result.success)
        queries = compute_queries(result.queries)
        all_metrics[f"{attack.name}_asr"] = asr
        for k, v in queries.items():
            all_metrics[f"{attack.name}_queries_{k}"] = float(v)

        # Quality metrics (if configured)
        quality_metrics = cfg.get("metrics", {}).get("quality", [])
        if quality_metrics and "lpips" in quality_metrics:
            lpips_score = compute_lpips(clean, result.adversarial)
            all_metrics[f"{attack.name}_lpips"] = lpips_score

        if quality_metrics and "clip_score" in quality_metrics:
            prompt = attack_cfg.get("prompt", "a photo")
            clip_s = compute_clip_score(result.adversarial, prompt)
            all_metrics[f"{attack.name}_clip_score"] = clip_s

        # Save adversarial samples directory (placeholder for future use)
        adv_dir = output_dir / "samples" / attack.name
        adv_dir.mkdir(exist_ok=True)

        for defense, defense_cfg in defense_methods:
            logger.info("  Applying defense: %s", defense.name)
            d_result: DefenseResult = defense.apply(result.adversarial, defense_cfg)

            ra = compute_robust_accuracy(model, d_result.defended, labels)
            cad = compute_clean_accuracy_drop(model, clean, d_result.defended, labels)
            lat = compute_latency([d_result.latency_sec])

            prefix = f"{attack.name}_vs_{defense.name}"
            all_metrics[f"{prefix}_robust_accuracy"] = ra
            all_metrics[f"{prefix}_clean_accuracy_drop"] = cad
            all_metrics[f"{prefix}_latency_mean"] = lat["mean"]

    # ------------------------------------------------------------------
    # 7. Generate sample grid
    # ------------------------------------------------------------------
    if attack_methods:
        first_attack, first_attack_cfg = attack_methods[0]
        first_result = first_attack.generate(clean, labels, model, first_attack_cfg)

        # Use first defense result if available
        defended_sample = None
        if defense_methods:
            first_defense, first_defense_cfg = defense_methods[0]
            d_result = first_defense.apply(first_result.adversarial, first_defense_cfg)
            defended_sample = d_result.defended

        grid_path = output_dir / "figures" / "sample_grid.png"
        generate_sample_grid(
            clean[:8],
            first_result.adversarial[:8],
            defended_sample[:8] if defended_sample is not None else None,
            save_path=grid_path,
        )
        logger.info("Saved sample grid to %s", grid_path)

    # ------------------------------------------------------------------
    # 8. Persist metrics
    # ------------------------------------------------------------------
    save_json(all_metrics, output_dir / "metrics.json")
    save_csv(all_metrics, output_dir / "metrics.csv")
    logger.info("Saved metrics to %s", output_dir)

    return output_dir
