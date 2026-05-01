"""Experiment orchestration pipeline.

Loads a YAML configuration, runs attacks and defenses against a target
model, computes evaluation metrics, and saves all artifacts to an output
directory.
"""

from __future__ import annotations

import logging
import csv
import statistics
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


def _resolve_seeds(cfg: DictConfig) -> list[int]:
    """Resolve the list of random seeds from config.

    Supports both ``task.seeds`` (list) and ``task.seed`` (single int) for
    backward compatibility. If neither is provided, defaults to ``[42]``.
    """
    if "seeds" in cfg.task and cfg.task.seeds is not None:
        seeds = OmegaConf.to_container(cfg.task.seeds, resolve=True)
        if isinstance(seeds, list):
            return [int(s) for s in seeds]
    if "seed" in cfg.task and cfg.task.seed is not None:
        return [int(cfg.task.seed)]
    return [42]


def _build_attack_defense_instances(
    cfg: DictConfig,
) -> tuple[list[tuple[Attack, dict]], list[tuple[Defense, dict]]]:
    """Instantiate attack and defense objects from config."""
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

    return attack_methods, defense_methods


def _run_attack_defense_pipeline(
    cfg: DictConfig,
    clean: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    seed: int,
    save_samples: bool = False,
) -> tuple[dict[str, float], dict, dict, list[tuple[AttackResult, dict]]]:
    """Run attack and defense evaluation pipeline for a single random seed.

    Args:
        cfg: Experiment configuration.
        clean: Clean images tensor (B, C, H, W) in [0, 1].
        labels: Ground-truth labels tensor (B,).
        model: Target classifier model.
        device: Compute device.
        output_dir: Artifacts output directory.
        seed: Random seed for this run.
        save_samples: If True, save per-attack and per-defense PNG samples.

    Returns:
        Tuple of (all_metrics, structured_attacks, structured_defenses, attack_results).
    """
    seed_everything(seed)

    attack_methods, defense_methods = _build_attack_defense_instances(cfg)

    all_metrics: dict[str, float] = {}
    structured_attacks: dict[str, dict] = {}
    structured_defenses: dict[str, dict] = {}
    attack_results: list[tuple[AttackResult, dict]] = []

    for attack, attack_cfg in attack_methods:
        # Text attacks operate on different inputs; skip in image pipeline
        if isinstance(attack, TextAttack):
            logger.info("Skipping text attack %s in image pipeline (seed=%d)", attack.name, seed)
            continue

        logger.info("Running attack: %s (seed=%d)", attack.name, seed)
        result: AttackResult = attack.generate(clean, labels, model, attack_cfg)
        attack_results.append((result, attack_cfg))

        # Attack-level metrics
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
        if save_samples:
            adv_dir = output_dir / "samples" / "adversarial" / attack.name
            adv_dir.mkdir(parents=True, exist_ok=True)
            _save_sample_pngs(result.adversarial, adv_dir, labels)

        for defense, defense_cfg in defense_methods:
            logger.info("  Applying defense: %s (seed=%d)", defense.name, seed)
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

            clean_defended_acc = compute_robust_accuracy(model, d_clean_result.defended, labels)
            all_metrics[f"{prefix}_clean_defended_accuracy"] = clean_defended_acc
            all_metrics[f"{prefix}_clean_latency_mean"] = clean_lat["mean"]

            structured_defenses[prefix] = {
                "robust_accuracy": ra,
                "clean_accuracy_drop": cad,
                "clean_defended_accuracy": clean_defended_acc,
                "latency_mean": lat["mean"],
                "latency_median": lat["median"],
                "latency_max": lat["max"],
                "clean_latency_mean": clean_lat["mean"],
                "clean_latency_median": clean_lat["median"],
                "clean_latency_max": clean_lat["max"],
                "metadata": d_result.metadata,
                "clean_metadata": d_clean_result.metadata,
            }

            # Save purified samples
            if save_samples:
                pur_dir = output_dir / "samples" / "purified" / f"{attack.name}_{defense.name}"
                pur_dir.mkdir(parents=True, exist_ok=True)
                _save_sample_pngs(d_result.defended, pur_dir, labels)

    return all_metrics, structured_attacks, structured_defenses, attack_results


def _aggregate_metrics(
    per_seed_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Compute mean and std across per-seed metric dicts.

    Returns a flat dict with ``_mean`` and ``_std`` suffixes for each metric
    key that appeared in at least one seed.
    """
    if not per_seed_metrics:
        return {}

    all_keys: list[str] = []
    seen: set[str] = set()
    for m in per_seed_metrics:
        for k in m:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    aggregated: dict[str, float] = {}
    for key in all_keys:
        values = [m[key] for m in per_seed_metrics if key in m]
        if len(values) >= 2:
            aggregated[f"{key}_mean"] = float(statistics.mean(values))
            aggregated[f"{key}_std"] = float(statistics.stdev(values))
        elif len(values) == 1:
            aggregated[f"{key}_mean"] = float(values[0])
            aggregated[f"{key}_std"] = 0.0

    return aggregated


def _aggregate_structured(
    per_seed_structured: list[dict],
    section: str,
) -> dict[str, dict]:
    """Aggregate structured attack or defense metrics across seeds.

    Returns a dict mapping attack/defense key to a dict of ``{metric: {mean, std}}``.
    """
    if not per_seed_structured:
        return {}

    # Collect all keys and their metric names
    all_keys: list[str] = []
    seen_keys: set[str] = set()
    metric_sets: dict[str, set[str]] = {}
    for s in per_seed_structured:
        section_data = s.get(section, {})
        for key, info in section_data.items():
            if key not in seen_keys:
                all_keys.append(key)
                seen_keys.add(key)
                metric_sets[key] = set()
            for mk in info:
                if isinstance(info[mk], (int, float)):
                    metric_sets[key].add(mk)

    aggregated: dict[str, dict] = {}
    for key in all_keys:
        entry: dict = {}
        for mk in sorted(metric_sets.get(key, set())):
            values = []
            for s in per_seed_structured:
                section_data = s.get(section, {})
                info = section_data.get(key, {})
                if mk in info and isinstance(info[mk], (int, float)):
                    values.append(float(info[mk]))
            if len(values) >= 2:
                entry[f"{mk}_mean"] = float(statistics.mean(values))
                entry[f"{mk}_std"] = float(statistics.stdev(values))
            elif len(values) == 1:
                entry[f"{mk}_mean"] = float(values[0])
                entry[f"{mk}_std"] = 0.0

        # Preserve non-numeric fields from first seed (e.g. queries dict, metadata)
        first = per_seed_structured[0].get(section, {}).get(key, {})
        for mk, mv in first.items():
            if not isinstance(mv, (int, float)):
                entry[mk] = mv

        aggregated[key] = entry

    return aggregated


def run_experiment(config_path: Path) -> Path:
    """Run a complete experiment from a YAML configuration file.

    Supports both single-seed (``task.seed``) and multi-seed (``task.seeds``)
    configurations. When multiple seeds are provided, the pipeline is executed
    once per seed and metrics are aggregated with mean and standard deviation.

    Pipeline:
        1. Load and snapshot configuration.
        2. Resolve seeds list.
        3. Load dataset and target model (shared across seeds).
        4. For each seed: run attack/defense pipeline.
        5. Aggregate metrics (mean ± std) across seeds.
        6. Generate charts and persist all artifacts.

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

    seeds = _resolve_seeds(cfg)
    logger.info("Loaded config from %s -> output %s (seeds=%s)", config_path, output_dir, seeds)

    # Snapshot the resolved config (store seeds list for reproducibility)
    snapshot_config(cfg, output_dir / "config.yaml")

    # ------------------------------------------------------------------
    # 2. Resolve device
    # ------------------------------------------------------------------
    device = get_device(cfg.task.device)
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 3. Load data (shared across seeds — deterministic subset)
    # ------------------------------------------------------------------
    dataset = _load_dataset(cfg.dataset)
    loader = DataLoader(dataset, batch_size=cfg.dataset.get("batch_size", 16), shuffle=False)
    clean, labels = _collect_tensor(loader, device)
    logger.info("Loaded %d samples", clean.shape[0])

    # ------------------------------------------------------------------
    # 4. Load target model (shared across seeds)
    # ------------------------------------------------------------------
    model = load_classifier(
        cfg.target_model.name,
        cfg.target_model.weights,
        device,
    )
    logger.info("Loaded model: %s (weights=%s)", cfg.target_model.name, cfg.target_model.weights)

    # ------------------------------------------------------------------
    # 5. Save clean samples (once)
    # ------------------------------------------------------------------
    clean_dir = output_dir / "samples" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    _save_sample_pngs(clean, clean_dir, labels)
    logger.info("Saved %d clean samples to %s", clean.shape[0], clean_dir)

    # ------------------------------------------------------------------
    # 6. Run attack/defense pipeline per seed
    # ------------------------------------------------------------------
    per_seed_metrics: list[dict[str, float]] = []
    per_seed_attacks: list[dict] = []
    per_seed_defenses: list[dict] = []
    first_seed_attack_results: list[tuple[AttackResult, dict]] = []

    for i, seed in enumerate(seeds):
        is_first = (i == 0)
        metrics, s_attacks, s_defenses, atk_results = _run_attack_defense_pipeline(
            cfg=cfg,
            clean=clean,
            labels=labels,
            model=model,
            device=device,
            output_dir=output_dir,
            seed=seed,
            save_samples=is_first,
        )
        per_seed_metrics.append(metrics)
        per_seed_attacks.append({"attacks": s_attacks})
        per_seed_defenses.append({"defenses": s_defenses})
        if is_first:
            first_seed_attack_results = atk_results
        logger.info("Completed seed %d (%d/%d)", seed, i + 1, len(seeds))

    # ------------------------------------------------------------------
    # 7. Aggregate metrics across seeds
    # ------------------------------------------------------------------
    aggregated_metrics = _aggregate_metrics(per_seed_metrics)
    aggregated_attacks = _aggregate_structured(per_seed_attacks, "attacks")
    aggregated_defenses = _aggregate_structured(per_seed_defenses, "defenses")

    # Also produce a flat dict with legacy-style keys (backward compat for
    # report generation and single-seed use). For single seed, use raw values
    # without _mean suffix. For multi-seed, keep both.
    if len(seeds) == 1:
        flat_metrics = per_seed_metrics[0]
    else:
        flat_metrics = dict(aggregated_metrics)

    # ------------------------------------------------------------------
    # 8. Generate charts
    # ------------------------------------------------------------------
    if first_seed_attack_results:
        first_result, _ = first_seed_attack_results[0]

        # Sample grid (from first seed)
        attack_methods, defense_methods = _build_attack_defense_instances(cfg)
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

        # Metric bars chart (from aggregated metrics, with error bars if multi-seed)
        attack_methods, _ = _build_attack_defense_instances(cfg)
        if len(attack_methods) > 1:
            bar_metrics_list = []
            bar_stds_list = []
            bar_labels = []
            has_any_std = False
            for atk, _ in attack_methods:
                if isinstance(atk, TextAttack):
                    continue
                atk_metrics = {}
                atk_stds = {}
                for mk in ["clean_accuracy", "adversarial_accuracy",
                           "asr_on_clean_correct", "prediction_change_rate"]:
                    key = f"{atk.name}_{mk}"
                    if len(seeds) == 1:
                        atk_metrics[mk] = flat_metrics.get(key, 0.0)
                    else:
                        atk_metrics[mk] = flat_metrics.get(f"{key}_mean", 0.0)
                        std_val = flat_metrics.get(f"{key}_std", 0.0)
                        if std_val > 0:
                            atk_stds[mk] = std_val
                            has_any_std = True
                if f"{atk.name}_clip_score" in flat_metrics or f"{atk.name}_clip_score_mean" in flat_metrics:
                    cs_key = f"{atk.name}_clip_score"
                    if len(seeds) == 1:
                        atk_metrics["clip_score"] = flat_metrics.get(cs_key, 0.0)
                    else:
                        atk_metrics["clip_score"] = flat_metrics.get(f"{cs_key}_mean", 0.0)
                        std_val = flat_metrics.get(f"{cs_key}_std", 0.0)
                        if std_val > 0:
                            atk_stds["clip_score"] = std_val
                            has_any_std = True
                if atk_metrics:
                    bar_metrics_list.append(atk_metrics)
                    bar_stds_list.append(atk_stds if has_any_std else None)
                    bar_labels.append(atk.name)
            if bar_metrics_list:
                bars_path = output_dir / "figures" / "metric_bars.png"
                generate_metric_bars(
                    bar_metrics_list,
                    bar_labels,
                    save_path=bars_path,
                    title="Attack Method Comparison",
                    stds_list=bar_stds_list if has_any_std else None,
                )

        # Radar chart (from aggregated metrics)
        radar_metrics = _build_radar_metrics(flat_metrics)
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
    # Save per-seed metrics for transparency
    per_seed_data = {}
    for seed, metrics in zip(seeds, per_seed_metrics):
        per_seed_data[str(seed)] = metrics
    save_json(per_seed_data, output_dir / "per_seed_metrics.json")

    # Save aggregated structured metrics
    aggregated_structured = {
        "attacks": aggregated_attacks,
        "defenses": aggregated_defenses,
    }
    save_json(aggregated_structured, output_dir / "structured_metrics.json")
    save_structured_csv(aggregated_structured, output_dir / "metrics_wide.csv")

    # Save flat aggregated metrics
    save_json(flat_metrics, output_dir / "metrics.json")
    save_csv(flat_metrics, output_dir / "metrics.csv")
    logger.info("Saved metrics to %s (seeds=%d)", output_dir, len(seeds))

    # ------------------------------------------------------------------
    # 10. Generate report
    # ------------------------------------------------------------------
    report_path = generate_report(output_dir)
    logger.info("Generated report: %s", report_path)

    # ------------------------------------------------------------------
    # 11. Release GPU memory before worker is reused (prevents fragmentation)
    # ------------------------------------------------------------------
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None:
            mps.empty_cache()

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

    Handles both single-seed keys (e.g. ``fgsm_asr_on_clean_correct``) and
    multi-seed aggregated keys (e.g. ``fgsm_asr_on_clean_correct_mean``).
    """
    if not metrics:
        return {}

    def _is_value(k: str, suffix: str) -> bool:
        """Match single-seed (k.endswith(suffix)) or multi-seed mean
        (k.endswith(suffix + '_mean')) keys, excluding std keys."""
        return (k.endswith(suffix) or k.endswith(suffix + "_mean")) and not k.endswith("_std")

    radar: dict[str, float] = {}

    asr_values = [
        float(v) for k, v in metrics.items()
        if _is_value(k, "_asr_on_clean_correct") and isinstance(v, (int, float))
    ]
    if not asr_values:
        asr_values = [
            float(v) for k, v in metrics.items()
            if _is_value(k, "_untargeted_asr") and isinstance(v, (int, float))
        ]
    if asr_values:
        radar["attack_resistance"] = _clamp01(1.0 - max(asr_values))

    robust_values = [
        float(v) for k, v in metrics.items()
        if _is_value(k, "_robust_accuracy") and isinstance(v, (int, float))
    ]
    if robust_values:
        radar["robust_accuracy"] = _clamp01(max(robust_values))

    drop_values = [
        float(v) for k, v in metrics.items()
        if _is_value(k, "_clean_accuracy_drop") and isinstance(v, (int, float))
    ]
    if drop_values:
        radar["clean_retention"] = _clamp01(1.0 - max(drop_values))

    lpips_values = [
        float(v) for k, v in metrics.items()
        if _is_value(k, "_lpips") and isinstance(v, (int, float))
    ]
    if lpips_values:
        radar["perceptual_similarity"] = _clamp01(1.0 - min(lpips_values))

    clip_values = [
        float(v) for k, v in metrics.items()
        if _is_value(k, "_clip_score") and isinstance(v, (int, float))
    ]
    if clip_values:
        radar["semantic_alignment"] = _clamp01(max(clip_values))

    latency_values = [
        float(v) for k, v in metrics.items()
        if _is_value(k, "_latency_mean") and isinstance(v, (int, float))
    ]
    if latency_values:
        radar["efficiency"] = _clamp01(1.0 - min(latency_values) / 10.0)

    return radar


def _clamp01(value: float) -> float:
    """Clamp a numeric value to the [0, 1] interval."""
    return max(0.0, min(1.0, value))
