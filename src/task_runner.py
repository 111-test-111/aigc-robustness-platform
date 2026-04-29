"""Experiment orchestration pipeline.

Loads a YAML configuration, runs attacks and defenses against a target
model, computes evaluation metrics, and saves all artifacts to an output
directory.
"""

from __future__ import annotations

import logging
import csv
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.attack_engine.adv_gan import AdvGANAttack
from src.attack_engine.base import Attack, AttackResult
from src.attack_engine.diffusion_attack import DiffusionAttack
from src.attack_engine.fgsm import FGSMAttack
from src.attack_engine.pgd import PGDAttack
from src.attack_engine.text_jailbreak import JailbreakAttack, TextAttack
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
from src.reporting.charts import generate_metric_bars, generate_radar, generate_sample_grid
from src.reporting.markdown_report import generate_report
from src.utils.device import get_device
from src.utils.io import save_csv, save_json, save_structured_csv, snapshot_config
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

ATTACK_REGISTRY: dict[str, type[Attack]] = {
    "fgsm": FGSMAttack,
    "pgd": PGDAttack,
    "advgan": AdvGANAttack,
    "diffusion": DiffusionAttack,
    "jailbreak": JailbreakAttack,  # type: ignore[dict-item]
}

DEFENSE_REGISTRY: dict[str, type[Defense]] = {
    "jpeg": JPEGDefense,
    "gaussian_blur": GaussianBlurDefense,
    "bit_depth": BitDepthDefense,
    "diffusion_purification": DiffusionPurificationDefense,
}


class ManifestImageDataset(Dataset):
    """Image dataset backed by a CSV manifest with explicit class indices.

    The manifest must contain either a header with ``path,label`` columns or
    headerless rows of ``relative/path.jpg,123``. Paths are resolved relative
    to ``root``.
    """

    def __init__(self, root: Path, manifest: Path, transform) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = self._read_manifest(Path(manifest))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        image_path = self.root / rel_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label

    def _read_manifest(self, manifest: Path) -> list[tuple[str, int]]:
        rows: list[tuple[str, int]] = []
        with open(manifest, newline="") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if first_row is None:
                raise ValueError(f"Empty manifest: {manifest}")

            def parse_row(row: list[str]) -> tuple[str, int] | None:
                if not row or len(row) < 2:
                    return None
                path_value = row[0].strip()
                label_value = row[1].strip()
                if path_value.lower() in {"path", "image", "relative_path"}:
                    return None
                return path_value, int(label_value)

            parsed = parse_row(first_row)
            if parsed is not None:
                rows.append(parsed)
            for row in reader:
                parsed = parse_row(row)
                if parsed is not None:
                    rows.append(parsed)

        if not rows:
            raise ValueError(f"Manifest contains no samples: {manifest}")
        return rows


def _load_dataset(cfg: DictConfig):
    """Load a dataset based on configuration.

    Supports:
    - ``cifar10``: CIFAR-10 via torchvision
    - ``image_folder``: Generic image folder (ImageFolder format)
    - ``imagenet_subset``: ImageNet validation subset
    - ``synthetic``: Random synthetic data for offline testing

    Args:
        cfg: Dataset section of the experiment config.  Must contain
            ``name``, ``root``, ``num_samples``, and ``image_size``.
            Optional: ``download`` (bool, default True for cifar10).

    Returns:
        A ``torch.utils.data.Subset`` or ``TensorDataset`` with at most
        ``num_samples`` items.

    Raises:
        ValueError: If the dataset name is not recognised.
    """
    download = cfg.get("download", True)
    num_samples = cfg.get("num_samples", 100)
    image_size = cfg.get("image_size", 32)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    if cfg.name == "cifar10":
        dataset = datasets.CIFAR10(
            root=cfg.root,
            train=False,
            download=download,
            transform=transform,
        )
    elif cfg.name == "image_folder":
        dataset = datasets.ImageFolder(
            root=cfg.root,
            transform=transform,
        )
    elif cfg.name == "imagenet_subset":
        manifest = cfg.get("manifest", None)
        if manifest:
            dataset = ManifestImageDataset(
                root=Path(cfg.root),
                manifest=Path(manifest),
                transform=transform,
            )
        else:
            dataset = datasets.ImageFolder(
                root=cfg.root,
                transform=transform,
            )
            if cfg.get("strict_labels", True):
                _validate_numeric_imagenet_folders(dataset)
    elif cfg.name == "synthetic":
        # Random synthetic data for offline testing (no disk/network access)
        images = torch.rand(num_samples, 3, image_size, image_size)
        labels = torch.randint(0, 10, (num_samples,))
        return torch.utils.data.TensorDataset(images, labels)
    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")

    indices = list(range(min(num_samples, len(dataset))))
    return Subset(dataset, indices)


def _validate_numeric_imagenet_folders(dataset: datasets.ImageFolder) -> None:
    """Require explicit numeric folder labels for ImageNet subset fallback."""
    non_numeric = [name for name in dataset.classes if not name.isdigit()]
    if non_numeric:
        preview = ", ".join(non_numeric[:5])
        raise ValueError(
            "imagenet_subset without a manifest requires class folders named "
            "with ImageNet class indices (0-999). Non-numeric folders found: "
            f"{preview}. Prefer dataset.manifest with path,label rows."
        )

    remapped_samples = []
    for path, class_index in dataset.samples:
        class_name = dataset.classes[class_index]
        label = int(class_name)
        if not 0 <= label <= 999:
            raise ValueError(f"ImageNet class index out of range: {label}")
        remapped_samples.append((path, label))
    dataset.samples = remapped_samples
    dataset.targets = [label for _, label in remapped_samples]


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
    loader = DataLoader(dataset, batch_size=cfg.dataset.get("batch_size", 16), shuffle=False)
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
    # 6. Save clean samples
    # ------------------------------------------------------------------
    clean_dir = output_dir / "samples" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    _save_sample_pngs(clean, clean_dir, labels)
    logger.info("Saved %d clean samples to %s", clean.shape[0], clean_dir)

    # ------------------------------------------------------------------
    # 7. Run attacks and defenses, collect metrics
    # ------------------------------------------------------------------
    all_metrics: dict[str, float] = {}
    # Structured metrics for report
    structured_attacks: dict[str, dict] = {}
    structured_defenses: dict[str, dict] = {}
    attack_results: list[tuple[AttackResult, dict]] = []

    for attack, attack_cfg in attack_methods:
        # Text attacks operate on different inputs; skip in image pipeline
        if isinstance(attack, TextAttack):
            logger.info("Skipping text attack %s in image pipeline", attack.name)
            continue

        logger.info("Running attack: %s", attack.name)
        result: AttackResult = attack.generate(clean, labels, model, attack_cfg)
        attack_results.append((result, attack_cfg))

        # Attack-level metrics. ``result.success`` is retained as the
        # implementation-defined attack success mask; for classifier attacks
        # we also compute stricter, label-aware metrics below.
        result_success_rate = compute_asr(result.success)
        queries = compute_queries(result.queries)
        all_metrics[f"{attack.name}_asr"] = result_success_rate
        for k, v in queries.items():
            all_metrics[f"{attack.name}_queries_{k}"] = float(v)

        # Extended attack metrics
        with torch.no_grad():
            clean_pred = model(clean).argmax(dim=1)
            adv_pred = model(result.adversarial).argmax(dim=1)
        clean_correct = clean_pred == labels
        adv_correct = adv_pred == labels
        clean_accuracy = float((clean_pred == labels).float().mean().item())
        adversarial_accuracy = float(adv_correct.float().mean().item())
        prediction_change_rate = float((adv_pred != clean_pred).float().mean().item())
        if clean_correct.any():
            asr_on_clean_correct = float(
                ((clean_correct) & (adv_pred != labels)).float().sum().item()
                / clean_correct.float().sum().item()
            )
        else:
            asr_on_clean_correct = 0.0

        target_class = attack_cfg.get("target_class")
        targeted_asr = None
        if target_class is not None:
            targeted_asr = float((adv_pred == int(target_class)).float().mean().item())

        # Keep the legacy key for compatibility, but make the more precise
        # metrics available for reporting and paper tables.
        untargeted_asr = asr_on_clean_correct
        all_metrics[f"{attack.name}_clean_accuracy"] = clean_accuracy
        all_metrics[f"{attack.name}_adversarial_accuracy"] = adversarial_accuracy
        all_metrics[f"{attack.name}_untargeted_asr"] = untargeted_asr
        all_metrics[f"{attack.name}_prediction_change_rate"] = prediction_change_rate
        all_metrics[f"{attack.name}_asr_on_clean_correct"] = asr_on_clean_correct
        if targeted_asr is not None:
            all_metrics[f"{attack.name}_targeted_asr"] = targeted_asr

        attack_info = {
            "asr": result_success_rate,
            "untargeted_asr": untargeted_asr,
            "asr_on_clean_correct": asr_on_clean_correct,
            "prediction_change_rate": prediction_change_rate,
            "targeted_asr": targeted_asr,
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adversarial_accuracy,
            "queries": queries,
            "metadata": result.metadata,
        }

        # Quality metrics (if configured)
        quality_metrics = cfg.get("metrics", {}).get("quality", [])
        if quality_metrics and "lpips" in quality_metrics:
            lpips_score = compute_lpips(clean, result.adversarial)
            all_metrics[f"{attack.name}_lpips"] = lpips_score
            attack_info["lpips"] = lpips_score

        if quality_metrics and "clip_score" in quality_metrics:
            prompt = attack_cfg.get("prompt", "a photo")
            clip_s = compute_clip_score(result.adversarial, prompt)
            all_metrics[f"{attack.name}_clip_score"] = clip_s
            attack_info["clip_score"] = clip_s

        if quality_metrics and "fid" in quality_metrics:
            from src.evaluation.quality_metrics import compute_fid
            fid_score = compute_fid(clean, result.adversarial)
            all_metrics[f"{attack.name}_fid"] = fid_score
            attack_info["fid"] = fid_score

        structured_attacks[attack.name] = attack_info

        # Save adversarial samples
        adv_dir = output_dir / "samples" / "adversarial" / attack.name
        adv_dir.mkdir(parents=True, exist_ok=True)
        _save_sample_pngs(result.adversarial, adv_dir, labels)

        for defense, defense_cfg in defense_methods:
            logger.info("  Applying defense: %s", defense.name)
            d_result: DefenseResult = defense.apply(result.adversarial, defense_cfg)
            d_clean_result = defense.apply(clean, defense_cfg)

            ra = compute_robust_accuracy(model, d_result.defended, labels)
            cad = compute_clean_accuracy_drop(model, clean, d_clean_result.defended, labels)
            lat = compute_latency([d_result.latency_sec])
            clean_lat = compute_latency([d_clean_result.latency_sec])

            prefix = f"{attack.name}_vs_{defense.name}"
            all_metrics[f"{prefix}_robust_accuracy"] = ra
            all_metrics[f"{prefix}_clean_accuracy_drop"] = cad
            all_metrics[f"{prefix}_latency_mean"] = lat["mean"]

            # Clean defended accuracy (defense applied to clean samples)
            clean_defended_acc = compute_robust_accuracy(model, d_clean_result.defended, labels)
            all_metrics[f"{prefix}_clean_defended_accuracy"] = clean_defended_acc
            all_metrics[f"{prefix}_clean_latency_mean"] = clean_lat["mean"]

            structured_defenses[prefix] = {
                "robust_accuracy": ra,
                "clean_accuracy_drop": cad,
                "clean_defended_accuracy": clean_defended_acc,
                "latency": lat,
                "clean_latency": clean_lat,
                "metadata": d_result.metadata,
                "clean_metadata": d_clean_result.metadata,
            }

            # Save purified samples
            pur_dir = output_dir / "samples" / "purified" / f"{attack.name}_{defense.name}"
            pur_dir.mkdir(parents=True, exist_ok=True)
            _save_sample_pngs(d_result.defended, pur_dir, labels)

    # ------------------------------------------------------------------
    # 8. Generate charts
    # ------------------------------------------------------------------
    if attack_results:
        first_result, _ = attack_results[0]

        # Sample grid
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

        # Metric bars chart
        if len(attack_methods) > 1:
            bar_metrics_list = []
            bar_labels = []
            for atk, _ in attack_methods:
                if isinstance(atk, TextAttack):
                    continue
                atk_metrics = {
                    "clean_acc": all_metrics.get(f"{atk.name}_clean_accuracy", 0.0),
                    "adv_acc": all_metrics.get(f"{atk.name}_adversarial_accuracy", 0.0),
                    "asr_clean_correct": all_metrics.get(
                        f"{atk.name}_asr_on_clean_correct", 0.0
                    ),
                    "pred_change": all_metrics.get(
                        f"{atk.name}_prediction_change_rate", 0.0
                    ),
                }
                if f"{atk.name}_clip_score" in all_metrics:
                    atk_metrics["clip_score"] = all_metrics[f"{atk.name}_clip_score"]
                if atk_metrics:
                    bar_metrics_list.append(atk_metrics)
                    bar_labels.append(atk.name)
            if bar_metrics_list:
                bars_path = output_dir / "figures" / "metric_bars.png"
                generate_metric_bars(
                    bar_metrics_list,
                    bar_labels,
                    save_path=bars_path,
                    title="Attack Method Comparison",
                )

        # Radar chart for first attack
        radar_metrics = _build_radar_metrics(all_metrics)
        if radar_metrics:
            radar_path = output_dir / "figures" / "radar.png"
            generate_radar(
                radar_metrics,
                save_path=radar_path,
                title="Robustness Radar",
            )

    # ------------------------------------------------------------------
    # 9. Persist metrics
    # ------------------------------------------------------------------
    # Save structured metrics separately for the report
    structured_metrics = {
        "attacks": structured_attacks,
        "defenses": structured_defenses,
    }
    save_json(structured_metrics, output_dir / "structured_metrics.json")
    save_structured_csv(structured_metrics, output_dir / "metrics_wide.csv")

    save_json(all_metrics, output_dir / "metrics.json")
    save_csv(all_metrics, output_dir / "metrics.csv")
    logger.info("Saved metrics to %s", output_dir)

    # ------------------------------------------------------------------
    # 10. Generate report
    # ------------------------------------------------------------------
    report_path = generate_report(output_dir)
    logger.info("Generated report: %s", report_path)

    return output_dir


def _save_sample_pngs(
    images: torch.Tensor,
    save_dir: Path,
    labels: torch.Tensor | None = None,
    max_samples: int = 16,
) -> None:
    """Save tensor images as individual PNG files.

    Args:
        images: (B, C, H, W) images in [0, 1]
        save_dir: Directory to save PNG files
        labels: Optional (B,) labels for naming
        max_samples: Maximum number of samples to save
    """
    n = min(images.shape[0], max_samples)
    for i in range(n):
        img_tensor = images[i].clamp(0, 1).cpu()
        img_pil = TF.to_pil_image(img_tensor)
        label_str = f"_label{labels[i].item()}" if labels is not None else ""
        img_pil.save(save_dir / f"sample_{i:04d}{label_str}.png")


def _build_radar_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Build normalized, higher-is-better radar dimensions.

    Raw metrics have mixed semantics: ASR and LPIPS are better when lower,
    while robust accuracy and CLIP score are better when higher. This helper
    converts them into comparable [0, 1] dimensions for visualization.
    """
    if not metrics:
        return {}

    radar: dict[str, float] = {}

    asr_values = [
        float(v)
        for k, v in metrics.items()
        if k.endswith("_asr_on_clean_correct") and isinstance(v, (int, float))
    ]
    if not asr_values:
        asr_values = [
            float(v)
            for k, v in metrics.items()
            if k.endswith("_untargeted_asr") and isinstance(v, (int, float))
        ]
    if asr_values:
        radar["attack_resistance"] = _clamp01(1.0 - max(asr_values))

    robust_values = [
        float(v)
        for k, v in metrics.items()
        if k.endswith("_robust_accuracy") and isinstance(v, (int, float))
    ]
    if robust_values:
        radar["robust_accuracy"] = _clamp01(max(robust_values))

    drop_values = [
        float(v)
        for k, v in metrics.items()
        if k.endswith("_clean_accuracy_drop") and isinstance(v, (int, float))
    ]
    if drop_values:
        radar["clean_retention"] = _clamp01(1.0 - max(drop_values))

    lpips_values = [
        float(v)
        for k, v in metrics.items()
        if k.endswith("_lpips") and isinstance(v, (int, float))
    ]
    if lpips_values:
        radar["perceptual_similarity"] = _clamp01(1.0 - min(lpips_values))

    clip_values = [
        float(v)
        for k, v in metrics.items()
        if k.endswith("_clip_score") and isinstance(v, (int, float))
    ]
    if clip_values:
        radar["semantic_alignment"] = _clamp01(max(clip_values))

    latency_values = [
        float(v)
        for k, v in metrics.items()
        if k.endswith("_latency_mean") and isinstance(v, (int, float))
    ]
    if latency_values:
        radar["efficiency"] = _clamp01(1.0 - min(latency_values) / 10.0)

    return radar


def _clamp01(value: float) -> float:
    """Clamp a numeric value to the [0, 1] interval."""
    return max(0.0, min(1.0, value))
