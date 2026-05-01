from __future__ import annotations

import concurrent.futures
import json
import logging
import multiprocessing
import time
from pathlib import Path

import typer

app = typer.Typer(help="AIGC 无限制对抗样本攻防验证平台")
logger = logging.getLogger(__name__)


# ======================================================================
# Worker functions (module-level, required for multiprocessing pickle)
# ======================================================================


def _run_one_experiment(config_path: str) -> str:
    """Run a single experiment from a YAML config path in a worker process.

    Returns the output directory path as a string (must be pickle-safe).
    Uses whichever GPU is visible by default (CUDA_VISIBLE_DEVICES).
    """
    from src.task_runner import run_experiment

    try:
        result = run_experiment(Path(config_path))
        return str(result)
    finally:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_one_experiment_gpu(config_path: str, gpu_id: int) -> str:
    """Run a single experiment pinned to a specific GPU.

    Sets ``CUDA_VISIBLE_DEVICES`` **before** importing torch so the
    worker process only sees the assigned GPU.  Must be called in a
    fresh ``spawn`` worker for the env-var to take effect.

    Because per-GPU ``ProcessPoolExecutor`` pools are used, every
    worker in a pool is dedicated to one GPU.  ``CUDA_VISIBLE_DEVICES``
    stays consistent even when the worker process is reused for
    multiple tasks.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from src.task_runner import run_experiment

    logger.info("Worker pinned to GPU %d (CUDA_VISIBLE_DEVICES=%s)", gpu_id, gpu_id)

    try:
        result = run_experiment(Path(config_path))
        return str(result)
    finally:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _is_sd_config(config_path: Path) -> bool:
    """Return True if the config requires a Stable Diffusion pipeline.

    Classifies a config as "SD-heavy" when any attack or defense uses the
    ``sd`` backend (i.e. loads an actual SD model, not ``mock``).  This
    determines VRAM scheduling in parallel mode.

    Uses raw YAML parsing for speed and to avoid pulling in OmegaConf at
    module-import time.  Falls back to OmegaConf when PyYAML is absent
    (unlikely, since OmegaConf itself depends on PyYAML).
    """
    try:
        import yaml

        with open(config_path) as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(config_path)
        except Exception:
            return False

    for attack in cfg.get("attacks", []):
        name = attack.get("name", "")
        backend = attack.get("backend", "")
        if name == "diffusion" and backend != "mock":
            return True
        if backend == "sd":
            return True

    for defense in cfg.get("defenses", []):
        name = defense.get("name", "")
        backend = defense.get("backend", "")
        if name == "diffusion_purification" and backend != "mock":
            return True
        if backend == "sd":
            return True

    return False


# ======================================================================
# CLI Commands
# ======================================================================


@app.command()
def run(
    config: Path = typer.Argument(..., help="实验配置 YAML 路径", exists=True),
) -> None:
    """运行攻防验证实验"""
    from src.task_runner import run_experiment

    typer.echo(f"加载配置: {config}")
    output_dir = run_experiment(config)
    typer.echo(f"实验完成，结果保存至: {output_dir}")


@app.command()
def run_batch(
    config_dir: Path = typer.Argument(..., help="包含 YAML 配置文件的目录"),
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="对比报告输出目录"
    ),
    parallel: bool = typer.Option(
        False, "--parallel", help="启用多进程并行运行实验"
    ),
    workers: int = typer.Option(
        None, "--workers", "-w", help="最大并行进程数 (默认自动检测)"
    ),
    sd_workers: int = typer.Option(
        None,
        "--sd-workers",
        help="SD 扩散类实验最大并行数 (默认自动检测)",
    ),
    gpus: str = typer.Option(
        "0",
        "--gpus",
        help="并行模式使用的 GPU 编号，逗号分隔 (例: '0,1')。每 GPU 独立分配进程",
    ),
) -> None:
    """批量运行目录中的所有实验并生成对比报告。

    \b
    顺序模式 (默认):
      所有实验在当前进程中逐个运行。SD pipeline 有进程级缓存，
      多个扩散实验顺序跑时 SD 仅加载一次。

    \b
    并行模式 (--parallel):
      两阶段 VRAM 感知调度，支持多 GPU 亲和性分配：
      - 阶段 1：SD 扩散类实验，每 GPU 最多 3 进程 (--sd-workers 可覆盖)
      - 阶段 2：轻量实验，每 GPU 最多 8 进程 (--workers 可覆盖)
      - 进程通过 CUDA_VISIBLE_DEVICES 绑定到指定 GPU，轮询分配

    \b
    示例:
      # 单卡 (默认 GPU 0)
      python -m src.cli run-batch configs/ --parallel

      # 双卡 5090 (推荐)
      python -m src.cli run-batch configs/ --parallel --gpus 0,1

      # 双卡 + 手动调参
      python -m src.cli run-batch configs/ --parallel --gpus 0,1 --sd-workers 8 --workers 20
    """
    if not config_dir.is_dir():
        typer.echo(f"目录不存在: {config_dir}")
        raise typer.Exit(1)

    configs = sorted(config_dir.rglob("*.yaml")) + sorted(config_dir.rglob("*.yml"))
    if not configs:
        typer.echo(f"未找到配置文件: {config_dir}")
        raise typer.Exit(1)

    typer.echo(f"找到 {len(configs)} 个配置文件")

    if parallel:
        gpu_list = [int(g.strip()) for g in gpus.split(",") if g.strip()]
        _run_batch_parallel(configs, output_dir, workers, sd_workers, gpu_list)
    else:
        _run_batch_sequential(configs, output_dir)


# ======================================================================
# Sequential runner
# ======================================================================


def _run_batch_sequential(
    configs: list[Path],
    output_dir: Path | None,
) -> None:
    """Run experiments one-by-one in the current process."""
    from src.task_runner import run_experiment

    report_dirs: list[Path] = []
    errors: list[tuple[str, str]] = []
    total = len(configs)

    t0 = time.monotonic()
    for i, cfg_path in enumerate(configs, 1):
        name = cfg_path.stem
        typer.echo(f"[{i}/{total}] {name} ... ", nl=False)
        try:
            t1 = time.monotonic()
            result = run_experiment(cfg_path)
            elapsed = time.monotonic() - t1
            report_dirs.append(result)
            typer.echo(f"OK ({elapsed:.0f}s)")
        except Exception as exc:
            errors.append((name, str(exc)))
            typer.echo(f"FAIL: {exc}")

    total_elapsed = time.monotonic() - t0
    _report_batch_summary(report_dirs, errors, total_elapsed, output_dir)


# ======================================================================
# Parallel runner (two-phase VRAM-aware scheduling)
# ======================================================================


def _run_batch_parallel(
    configs: list[Path],
    output_dir: Path | None,
    max_workers: int | None,
    max_sd_workers: int | None,
    gpu_list: list[int],
) -> None:
    """Run experiments with VRAM-aware two-phase scheduling + GPU affinity.

    Phase 1 — SD-heavy configs with limited concurrency.
    Phase 2 — Light configs with full concurrency.

    Workers are pinned to GPUs via ``CUDA_VISIBLE_DEVICES`` in round-robin
    order.  Each phase uses a fresh ProcessPoolExecutor so failed workers
    do not contaminate subsequent work.
    """
    num_gpus = len(gpu_list)

    # Sensible defaults that scale with GPU count.
    # SD 1.5 + classifier ≈ 7-9 GB VRAM → 3 per 32 GB GPU is safe.
    # Light configs (classifier only) ≈ 1-2 GB → 8 per GPU.
    max_sd_workers = max_sd_workers or (num_gpus * 3)
    max_workers = max_workers or (num_gpus * 8)

    # Classify configs
    sd_configs = [c for c in configs if _is_sd_config(c)]
    sd_set = set(sd_configs)
    light_configs = [c for c in configs if c not in sd_set]

    typer.echo(
        f"GPUs: {gpu_list}  |  "
        f"SD-heavy: {len(sd_configs)}  Light: {len(light_configs)}  |  "
        f"Workers: SD={max_sd_workers}  Light={max_workers}"
    )
    typer.echo("")

    report_dirs: list[Path] = []
    errors: list[tuple[str, str]] = []

    # Spawn context is required for CUDA safety (fork + CUDA = undefined behaviour)
    ctx = multiprocessing.get_context("spawn")

    t0 = time.monotonic()

    # --- Phase 1: SD-heavy ---------------------------------------------------
    if sd_configs:
        actual_workers = min(max_sd_workers, len(sd_configs))
        typer.echo(
            f"── Phase 1: SD-heavy ({len(sd_configs)} configs, "
            f"{actual_workers} workers on {num_gpus} GPU(s)) ──"
        )
        _run_parallel_phase(
            sd_configs, max_sd_workers, ctx, report_dirs, errors, gpu_list
        )

    # --- Phase 2: Light ------------------------------------------------------
    if light_configs:
        actual_workers = min(max_workers, len(light_configs))
        typer.echo(
            f"── Phase 2: Light ({len(light_configs)} configs, "
            f"{actual_workers} workers on {num_gpus} GPU(s)) ──"
        )
        _run_parallel_phase(
            light_configs, max_workers, ctx, report_dirs, errors, gpu_list
        )

    total_elapsed = time.monotonic() - t0
    _report_batch_summary(report_dirs, errors, total_elapsed, output_dir)


def _run_parallel_phase(
    configs: list[Path],
    total_workers: int,
    mp_context,
    report_dirs: list[Path],
    errors: list[tuple[str, str]],
    gpu_list: list[int],
) -> None:
    """Execute a batch of configs in parallel with per-GPU process pools.

    **Why per-GPU pools (not one pool with round-robin GPU assignment):**
    ``ProcessPoolExecutor`` reuses worker processes across tasks.  If a
    single pool with mixed GPU assignments were used, a worker that ran a
    GPU-0 task (and imported torch) could later be handed a GPU-1 task.
    At that point ``CUDA_VISIBLE_DEVICES`` is ineffective because torch
    is already initialized — the task would silently run on the wrong GPU.

    By creating one pool per GPU, every worker in a pool is dedicated to
    that GPU for its entire lifetime.  ``CUDA_VISIBLE_DEVICES`` is set
    once (on first ``import torch``) and stays correct.

    Crash isolation: if one GPU's pool dies (OOM, driver crash), the
    error is logged and remaining GPUs continue unaffected.
    """
    num_gpus = len(gpu_list)
    workers_per_gpu = max(total_workers // num_gpus, 1)
    total = len(configs)

    # Single-GPU path: no env-var dance needed.
    if num_gpus == 1:
        _run_parallel_phase_single_gpu(
            configs, total_workers, mp_context, report_dirs, errors
        )
        return

    # Multi-GPU path: one pool per GPU.
    # Build pools first, then distribute configs round-robin.
    pools: list[tuple[concurrent.futures.ProcessPoolExecutor, int]] = []
    for gpu_id in gpu_list:
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers_per_gpu,
            mp_context=mp_context,
        )
        pools.append((pool, gpu_id))

    try:
        futures: dict[concurrent.futures.Future, tuple[Path, int]] = {}
        for i, cfg_path in enumerate(configs):
            pool, gpu_id = pools[i % num_gpus]
            future = pool.submit(_run_one_experiment_gpu, str(cfg_path), gpu_id)
            futures[future] = (cfg_path, gpu_id)

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            cfg_path, gpu_id = futures[future]
            completed += 1
            try:
                result_path = future.result()
                report_dirs.append(Path(result_path))
                typer.echo(
                    f"  [{completed}/{total}] OK  {cfg_path.stem}  (GPU {gpu_id})"
                )
            except concurrent.futures.process.BrokenProcessPool:
                errors.append(
                    (cfg_path.stem, f"Worker crashed on GPU {gpu_id} (likely OOM)")
                )
                typer.echo(
                    f"  [{completed}/{total}] CRASH  {cfg_path.stem}  "
                    f"(GPU {gpu_id} pool dead)"
                )
                # Don't raise — other GPU pools may still be healthy.
                # Mark remaining futures for this GPU as failed.
                for f, (p, g) in list(futures.items()):
                    if g == gpu_id and not f.done():
                        errors.append((p.stem, f"Skipped (GPU {gpu_id} pool dead)"))
                        futures.pop(f, None)
            except Exception as exc:
                errors.append((cfg_path.stem, str(exc)))
                typer.echo(
                    f"  [{completed}/{total}] FAIL  {cfg_path.stem}: {exc}"
                )
    finally:
        for pool, _ in pools:
            pool.shutdown(wait=False, cancel_futures=True)


def _run_parallel_phase_single_gpu(
    configs: list[Path],
    workers: int,
    mp_context,
    report_dirs: list[Path],
    errors: list[tuple[str, str]],
) -> None:
    """Single-GPU fast path — no GPU pinning needed."""
    workers = min(workers, len(configs))
    total = len(configs)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
    ) as pool:
        futures: dict[concurrent.futures.Future, Path] = {}
        for cfg_path in configs:
            future = pool.submit(_run_one_experiment, str(cfg_path))
            futures[future] = cfg_path

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            cfg_path = futures[future]
            completed += 1
            try:
                result_path = future.result()
                report_dirs.append(Path(result_path))
                typer.echo(f"  [{completed}/{total}] OK  {cfg_path.stem}")
            except concurrent.futures.process.BrokenProcessPool:
                errors.append((cfg_path.stem, "Worker crashed (likely CUDA OOM)"))
                typer.echo(f"  [{completed}/{total}] CRASH  {cfg_path.stem}")
                raise
            except Exception as exc:
                errors.append((cfg_path.stem, str(exc)))
                typer.echo(f"  [{completed}/{total}] FAIL  {cfg_path.stem}: {exc}")


# ======================================================================
# Shared reporting
# ======================================================================


def _report_batch_summary(
    report_dirs: list[Path],
    errors: list[tuple[str, str]],
    total_elapsed: float,
    output_dir: Path | None,
) -> None:
    """Print a batch-run summary and generate a comparison report."""
    mins = total_elapsed / 60.0
    typer.echo("")
    typer.echo(
        f"── {len(report_dirs)} OK, {len(errors)} FAIL "
        f"({mins:.1f} min elapsed) ──"
    )

    if errors:
        typer.echo("")
        for name, err in errors:
            typer.echo(f"  FAIL  {name}: {err}")

    if not report_dirs:
        typer.echo("No successful experiments — skipping comparison report.")
        return

    comp_dir = output_dir or (report_dirs[0].parent / "comparison")
    _generate_comparison(report_dirs, comp_dir)
    typer.echo(f"对比报告: {comp_dir}")


# ======================================================================
# Comparison report generation (unchanged, kept for backward compat)
# ======================================================================


def _generate_comparison(report_dirs: list[Path], comp_dir: Path) -> None:
    """Generate comparison charts and report from multiple experiments."""
    from src.reporting.charts import generate_metric_bars, generate_radar
    from src.task_runner import _build_radar_metrics

    all_metrics: list[dict[str, float]] = []
    labels: list[str] = []
    all_structured: list[dict] = []

    for d in report_dirs:
        metrics_path = d / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            all_metrics.append(numeric_metrics)
            labels.append(d.name)

        structured_path = d / "structured_metrics.json"
        if structured_path.exists():
            all_structured.append(json.loads(structured_path.read_text()))

    if not all_metrics:
        return

    comp_dir.mkdir(parents=True, exist_ok=True)
    (comp_dir / "figures").mkdir(exist_ok=True)

    if all_metrics:
        generate_metric_bars(
            all_metrics,
            labels,
            save_path=comp_dir / "figures" / "comparison_bars.png",
            title="Experiment Comparison",
        )

    for i, (metrics, label) in enumerate(zip(all_metrics, labels)):
        radar_metrics = _build_radar_metrics(metrics)
        if radar_metrics:
            generate_radar(
                radar_metrics,
                save_path=comp_dir / "figures" / f"radar_{label}.png",
                title=f"{label} Robustness",
            )

    _generate_comparison_report(all_metrics, all_structured, labels, comp_dir)

    combined = dict(zip(labels, all_metrics))
    (comp_dir / "comparison.json").write_text(json.dumps(combined, indent=2))

    _save_comparison_csv(all_metrics, labels, comp_dir)


def _generate_comparison_report(
    all_metrics: list[dict[str, float]],
    all_structured: list[dict],
    labels: list[str],
    comp_dir: Path,
) -> None:
    """Generate a markdown comparison report."""
    from datetime import datetime, timezone

    lines = [
        "# Experiment Comparison Report",
        "",
        f"> Auto-generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## 1. Experiment List",
        "",
        "| No. | Experiment Name |",
        "|-----|-----------------|",
    ]

    for i, label in enumerate(labels, 1):
        lines.append(f"| {i} | {label} |")

    lines.extend(["", "## 2. Attack Method Comparison", ""])

    attack_names: set[str] = set()
    for structured in all_structured:
        attack_names.update(structured.get("attacks", {}).keys())

    if attack_names:
        lines.append("| Experiment | Attack Method | ASR | Clean Accuracy | Adversarial Accuracy |")
        lines.append("|------------|---------------|-----|----------------|----------------------|")
        for structured, label in zip(all_structured, labels):
            for attack_name, info in structured.get("attacks", {}).items():
                asr = f"{info.get('asr', 0) * 100:.1f}%"
                clean_acc = f"{info.get('clean_accuracy', 0) * 100:.1f}%"
                adv_acc = f"{info.get('adversarial_accuracy', 0) * 100:.1f}%"
                lines.append(f"| {label} | {attack_name} | {asr} | {clean_acc} | {adv_acc} |")

    lines.extend(["", "## 3. Defense Method Comparison", ""])

    defense_names: set[str] = set()
    for structured in all_structured:
        defense_names.update(structured.get("defenses", {}).keys())

    if defense_names:
        lines.append("| Experiment | Attack-Defense | Robust Accuracy | Clean Drop | Latency (s) |")
        lines.append("|------------|----------------|-----------------|------------|-------------|")
        for structured, label in zip(all_structured, labels):
            for defense_name, info in structured.get("defenses", {}).items():
                ra = f"{info.get('robust_accuracy', 0) * 100:.1f}%"
                cad = f"{info.get('clean_accuracy_drop', 0) * 100:.1f}%"
                lat = f"{info.get('latency', {}).get('mean', 0):.4f}"
                lines.append(f"| {label} | {defense_name} | {ra} | {cad} | {lat} |")

    lines.extend(["", "## 4. Visualizations", ""])
    lines.append("### Comparison Bar Chart")
    lines.append("")
    lines.append("![Comparison Bar Chart](figures/comparison_bars.png)")
    lines.append("")

    for label in labels:
        radar_path = f"figures/radar_{label}.png"
        lines.append(f"### {label} Radar Chart")
        lines.append("")
        lines.append(f"![{label} Radar Chart]({radar_path})")
        lines.append("")

    lines.extend([
        "---",
        "*Report auto-generated by AIGC Robustness Platform*",
    ])

    (comp_dir / "comparison.md").write_text("\n".join(lines))


def _save_comparison_csv(
    all_metrics: list[dict[str, float]],
    labels: list[str],
    comp_dir: Path,
) -> None:
    """Save comparison metrics as a wide CSV table."""
    import csv

    all_keys: list[str] = []
    seen: set[str] = set()
    for metrics in all_metrics:
        for k in metrics:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    csv_path = comp_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment"] + all_keys)
        for metrics, label in zip(all_metrics, labels):
            row = [label] + [metrics.get(k, "") for k in all_keys]
            writer.writerow(row)


# ======================================================================
# Other commands
# ======================================================================


@app.command()
def ui(
    model: str = "resnet50",
    device: str = "auto",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio web interface."""
    from src.web_ui import create_app

    application = create_app(model, device)
    application.launch(server_port=port, share=share)


@app.command()
def hello() -> None:
    """测试命令"""
    typer.echo("AIGC Robustness Platform")


if __name__ == "__main__":
    app()
