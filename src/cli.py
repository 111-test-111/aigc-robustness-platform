from __future__ import annotations

import atexit
import concurrent.futures
import json
import logging
import multiprocessing
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(help="AIGC 无限制对抗样本攻防验证平台")
logger = logging.getLogger(__name__)
DEFAULT_SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


# ======================================================================
# Signal handling — graceful shutdown on Ctrl+C
# ======================================================================

# Tracks every ProcessPoolExecutor created in this session so the SIGINT
# handler can shut them all down immediately, even if the interrupt
# arrives while the main loop is blocked in as_completed().
_active_pools: list[concurrent.futures.ProcessPoolExecutor] = []
_sigint_count = 0


def _register_pool(pool: concurrent.futures.ProcessPoolExecutor) -> None:
    _active_pools.append(pool)


def _unregister_pool(pool: concurrent.futures.ProcessPoolExecutor) -> None:
    try:
        _active_pools.remove(pool)
    except ValueError:
        pass


def _kill_orphan_workers() -> None:
    """SIGKILL any Python child processes whose command line mentions
    ``run_experiment`` and whose parent is this process.

    Called after pool shutdown to guarantee no GPU-leaking zombies.
    """
    current_pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(current_pid), "-f", "run_experiment"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            if not line:
                continue
            pid = int(line.strip())
            try:
                os.kill(pid, signal.SIGKILL)
                logger.info("Killed orphan worker PID %d", pid)
            except ProcessLookupError:
                pass
    except Exception:
        pass


def _force_cleanup() -> None:
    """Shutdown every tracked pool and kill any surviving workers.

    Registered with :func:`atexit` so even an unexpected fatal exit
    (e.g. unhandled exception) triggers cleanup.
    """
    for pool in list(_active_pools):
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    _active_pools.clear()
    _kill_orphan_workers()


atexit.register(_force_cleanup)


def _handle_sigint(signum: int, _frame: object) -> None:
    """First Ctrl+C → graceful shutdown.  Second → immediate ``os._exit``.

    Shuts down all tracked pools, kills orphaned workers, then raises
    ``KeyboardInterrupt`` so the normal ``finally`` blocks unwind the
    stack cleanly.
    """
    global _sigint_count
    _sigint_count += 1

    if _sigint_count >= 2:
        logger.warning("Second interrupt — forcing immediate exit")
        os._exit(1)

    logger.warning(
        "Interrupt received, shutting down worker pools… "
        "(press Ctrl+C again to force-exit)"
    )

    # Shut down every pool we know about *now*, before KeyboardInterrupt
    # propagates.  This keeps worker teardown as orderly as possible.
    for pool in list(_active_pools):
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    _active_pools.clear()

    # Brief grace period for workers to exit on their own.
    time.sleep(0.3)

    # Anything still alive gets SIGKILL.
    _kill_orphan_workers()

    raise KeyboardInterrupt


def _install_signal_handlers() -> None:
    """Wire SIGINT and SIGTERM to our graceful-shutdown handler."""
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)


# Install handlers at import time so they are active for every CLI command.
_install_signal_handlers()


# ======================================================================
# Worker functions (module-level, required for multiprocessing pickle)
# ======================================================================


def _setup_worker_env() -> None:
    """Configure environment before importing torch in a worker process.

    - Suppress known deprecation warnings from transformers / huggingface_hub
      that are emitted during Stable Diffusion pipeline loading.
    - Enable CUDA expandable memory segments to reduce fragmentation OOMs.
    """
    import warnings

    # transformers: Siglip2ImageProcessorFast → Siglip2ImageProcessor
    # transformers: safetensors missing → falling back to pickle
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")
    warnings.filterwarnings("ignore", message=".*allow_pickle.*")
    warnings.filterwarnings("ignore", message=".*unsafe serialization.*")
    # huggingface_hub: local_dir_use_symlinks is deprecated and ignored
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    # torchvision: lpips uses deprecated pretrained=True for AlexNet
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _empty_device_cache() -> None:
    """Release cached CUDA memory after an experiment finishes.

    Must be called **after** ``import torch`` (otherwise the CUDA backend
    isn't loaded yet and the call is a no-op).
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_one_experiment(config_path: str) -> str:
    """Run a single experiment from a YAML config path in a worker process.

    Returns the output directory path as a string (must be pickle-safe).
    Uses whichever GPU is visible by default (CUDA_VISIBLE_DEVICES).
    """
    _setup_worker_env()

    from src.task_runner import run_experiment

    try:
        result = run_experiment(Path(config_path))
        return str(result)
    finally:
        _empty_device_cache()


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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _setup_worker_env()

    from src.task_runner import run_experiment

    logger.info("Worker pinned to GPU %d (CUDA_VISIBLE_DEVICES=%s)", gpu_id, gpu_id)

    try:
        result = run_experiment(Path(config_path))
        return str(result)
    finally:
        _empty_device_cache()


def _is_sd_config(config_path: Path) -> bool:
    """Return True if the config requires a Stable Diffusion pipeline.

    Classifies a config as "SD-heavy" when any attack or defense uses the
    ``sd`` backend (i.e. loads an actual SD model, not ``mock``).  This
    determines VRAM scheduling in parallel mode.
    """
    cfg = _load_plain_config(config_path)
    return any(
        _uses_sd_backend(item, "diffusion")
        for item in _list_config_section(cfg.get("attacks", []))
    ) or any(
        _uses_sd_backend(item, "diffusion_purification")
        for item in _list_config_section(cfg.get("defenses", []))
    )


def _is_oom_error(exc: Exception) -> bool:
    """Return True if the exception is a CUDA out-of-memory error."""
    msg = str(exc).lower()
    return "out of memory" in msg


def _load_plain_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config as a plain dict for lightweight CLI inspection."""
    try:
        import yaml

        with open(config_path) as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        try:
            from omegaconf import OmegaConf

            data = OmegaConf.to_container(
                OmegaConf.load(config_path), resolve=True
            ) or {}
        except Exception:
            return {}

    return data if isinstance(data, dict) else {}


def _list_config_section(value: Any) -> list[dict[str, Any]]:
    """Return a list of dict entries from a config section."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _uses_sd_backend(item: dict[str, Any], default_sd_name: str) -> bool:
    """Return True when a config item will load a real SD pipeline."""
    name = str(item.get("name", ""))
    backend = str(item.get("backend", ""))
    return backend == "sd" or (name == default_sd_name and backend != "mock")


def _collect_prewarm_requirements(
    configs: list[Path],
) -> tuple[set[str], set[str], bool, bool, bool]:
    """Collect classifiers, SD pipelines, and quality models needed by configs."""
    classifiers: set[str] = set()
    sd_pipelines: set[str] = set()
    needs_lpips = False
    needs_fid = False
    needs_clip = False

    for cfg_path in configs:
        cfg = _load_plain_config(cfg_path)
        if not cfg:
            continue

        tm = cfg.get("target_model", {})
        if isinstance(tm, dict):
            weights = str(tm.get("weights", "imagenet")).lower()
            if weights != "none":
                classifiers.add(str(tm.get("name", "resnet50")))

        for attack in _list_config_section(cfg.get("attacks", [])):
            if _uses_sd_backend(attack, "diffusion"):
                model_id = (
                    attack.get("generator")
                    or attack.get("model_id")
                    or DEFAULT_SD_MODEL_ID
                )
                if model_id and model_id != "mock":
                    sd_pipelines.add(str(model_id))

        for defense in _list_config_section(cfg.get("defenses", [])):
            if _uses_sd_backend(defense, "diffusion_purification"):
                model_id = (
                    defense.get("model_id")
                    or defense.get("generator")
                    or DEFAULT_SD_MODEL_ID
                )
                if model_id and model_id != "mock":
                    sd_pipelines.add(str(model_id))

        metrics = cfg.get("metrics", {})
        quality = metrics.get("quality", []) if isinstance(metrics, dict) else []
        if isinstance(quality, list):
            needs_lpips = needs_lpips or "lpips" in quality
            needs_fid = needs_fid or "fid" in quality
            needs_clip = needs_clip or "clip_score" in quality

    return classifiers, sd_pipelines, needs_lpips, needs_fid, needs_clip


def _prewarm_lpips_weights() -> None:
    """Download and verify LPIPS AlexNet weights used by quality metrics."""
    import torch

    from src.evaluation.quality_metrics import compute_lpips

    img = torch.zeros(1, 3, 64, 64)
    compute_lpips(img, img)


def _prewarm_fid_weights() -> None:
    """Download and verify the Inception weights used by torchmetrics FID."""
    import torch

    weights_url = (
        "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
        "weights-inception-2015-12-05-6726825d.pth"
    )

    # torchmetrics delegates to torch-fidelity, whose Inception weights live
    # in torch.hub's checkpoint cache. Download that exact file up front so
    # workers do not race on GitHub during the first FID computation.
    torch.hub.load_state_dict_from_url(
        weights_url,
        map_location="cpu",
        progress=True,
    )

    from torchmetrics.image.fid import FrechetInceptionDistance

    # Cache warmup does not need GPU residency; keep it on CPU to avoid a
    # transient Inception allocation after SD has just been loaded and freed.
    fid_device = torch.device("cpu")
    fid = FrechetInceptionDistance(feature=64).to(fid_device)

    # Run a full real/fake/compute cycle. Some versions delay feature extractor
    # initialization until compute(), so update(real=True) alone is insufficient.
    real = torch.zeros(2, 3, 299, 299, dtype=torch.uint8, device=fid_device)
    fake = torch.full((2, 3, 299, 299), 255, dtype=torch.uint8, device=fid_device)
    fid.update(real, real=True)
    fid.update(fake, real=False)
    fid.compute()
    del fid, real, fake


def _prewarm_from_configs(configs: list[Path]) -> None:
    """Download & cache all models referenced by *configs* sequentially.

    Must be called **before** spawning worker processes so that concurrent
    workers find the models already on disk instead of racing to download
    them (which causes duplicate downloads, file-lock conflicts, and
    slow startup in parallel mode).
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"── 模型预热 (device={device}) ──")

    classifiers, sd_pipelines, needs_lpips, needs_fid, needs_clip = (
        _collect_prewarm_requirements(configs)
    )

    if (
        not classifiers
        and not sd_pipelines
        and not needs_lpips
        and not needs_fid
        and not needs_clip
    ):
        typer.echo("  No models to prewarm.")
        return

    failures: list[str] = []

    # Warm classifiers ---------------------------------------------------
    if classifiers:
        from src.model_zoo.classifiers import load_classifier

        for name in sorted(classifiers):
            typer.echo(f"  Loading classifier: {name} ... ", nl=False)
            try:
                model = load_classifier(name, "imagenet", device)
                del model
                typer.echo("OK")
            except Exception as exc:
                failures.append(f"classifier {name}: {exc}")
                typer.echo(f"FAIL ({exc})")

    # Warm SD pipelines --------------------------------------------------
    if sd_pipelines:
        from src.model_zoo.generators import _PIPELINE_CACHE, load_sd_pipeline

        for model_id in sorted(sd_pipelines):
            typer.echo(f"  Loading SD pipeline: {model_id}")
            try:
                dtype = torch.float16 if device.type == "cuda" else torch.float32
                pipe = load_sd_pipeline(
                    model_id=model_id,
                    device=device,
                    torch_dtype=dtype,
                )
                del pipe
                _PIPELINE_CACHE.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                typer.echo("    OK")
            except Exception as exc:
                failures.append(f"SD pipeline {model_id}: {exc}")
                typer.echo(f"    FAIL ({exc})")

    # Warm LPIPS weights -------------------------------------------------
    if needs_lpips:
        typer.echo("  Loading LPIPS model: alex ... ", nl=False)
        try:
            _prewarm_lpips_weights()
            typer.echo("OK")
        except Exception as exc:
            failures.append(f"LPIPS alex: {exc}")
            typer.echo(f"FAIL ({exc})")

    # Warm FID inception weights (torchmetrics delegates to torch-fidelity) ---
    if needs_fid:
        typer.echo("  Pre-downloading FID inception weights ... ", nl=False)
        try:
            _prewarm_fid_weights()
            typer.echo("OK")
        except Exception as exc:
            failures.append(f"FID inception weights: {exc}")
            typer.echo(f"FAIL ({exc})")

    # Warm CLIP (used by quality metrics) --------------------------------
    if needs_clip:
        typer.echo("  Loading CLIP model: openai/clip-vit-base-patch16 ... ", nl=False)
        try:
            from transformers import CLIPModel, CLIPProcessor

            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            del clip_model, clip_proc
            typer.echo("OK")
        except Exception as exc:
            failures.append(f"CLIP openai/clip-vit-base-patch16: {exc}")
            typer.echo(f"FAIL ({exc})")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if failures:
        typer.echo("── 预热失败：以下资源没有成功缓存 ──")
        for item in failures:
            typer.echo(f"  - {item}")
        raise typer.Exit(1)

    typer.echo("── 预热完成 ──\n")


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
    no_prewarm: bool = typer.Option(
        False,
        "--no-prewarm",
        help="跳过模型预热 (默认在批量运行前自动预热)",
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
      - 默认自动预热：运行前顺序加载所有模型到本地缓存，
        避免下载超时和并发冲突
      - 阶段 1：SD 扩散类实验，每 GPU 最多 1 进程 (--sd-workers 可覆盖)
      - 阶段 2：轻量实验，每 GPU 最多 8 进程 (--workers 可覆盖)
      - 进程通过 CUDA_VISIBLE_DEVICES 绑定到指定 GPU，轮询分配
      - OOM 失败的实验会在阶段结束后自动串行重试（独占 GPU 资源）
      - Ctrl+C 优雅终止：第一次中断立即关闭所有 worker 池并清理残留进程
      - 使用 --no-prewarm 跳过预热（适用于模型已缓存的重复运行）

    \b
    示例:
      # 单卡 (默认 GPU 0)
      python -m src.cli run-batch configs/ --parallel

      # 双卡 5090 (推荐)
      python -m src.cli run-batch configs/ --parallel --gpus 0,1

      # 双卡 + 手动调参 (想提高 SD 并发时使用)
      python -m src.cli run-batch configs/ --parallel --gpus 0,1 --sd-workers 4 --workers 20
    """
    if not config_dir.is_dir():
        typer.echo(f"目录不存在: {config_dir}")
        raise typer.Exit(1)

    configs = sorted(config_dir.rglob("*.yaml")) + sorted(config_dir.rglob("*.yml"))
    if not configs:
        typer.echo(f"未找到配置文件: {config_dir}")
        raise typer.Exit(1)

    typer.echo(f"找到 {len(configs)} 个配置文件")

    if not no_prewarm:
        _prewarm_from_configs(configs)

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
# Parallel runner (two-phase VRAM-aware scheduling + OOM retry)
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
    Retry  — OOM failures are retried sequentially with dedicated GPUs.

    Workers are pinned to GPUs via ``CUDA_VISIBLE_DEVICES`` in round-robin
    order.  Each phase uses a fresh ProcessPoolExecutor so failed workers
    do not contaminate subsequent work.
    """
    num_gpus = len(gpu_list)

    # SD 1.5 + classifier at 512×512 ≈ 10-15 GB peak VRAM per process;
    # heavy configs with attack+defence can spike to 20 GB.
    # 1 per 32 GB GPU is the safe default; use --sd-workers to override.
    # Light configs (classifier only) ≈ 1-2 GB → 8 per GPU.
    max_sd_workers = max_sd_workers or (num_gpus * 1)
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
    oom_retry: list[Path] = []

    try:
        # --- Phase 1: SD-heavy -----------------------------------------------
        if sd_configs:
            actual_workers = min(max_sd_workers, len(sd_configs))
            typer.echo(
                f"── Phase 1: SD-heavy ({len(sd_configs)} configs, "
                f"{actual_workers} workers on {num_gpus} GPU(s)) ──"
            )
            oom = _run_parallel_phase(
                sd_configs, max_sd_workers, ctx, report_dirs, errors, gpu_list
            )
            oom_retry.extend(oom)

        # --- Phase 2: Light --------------------------------------------------
        if light_configs:
            actual_workers = min(max_workers, len(light_configs))
            typer.echo(
                f"── Phase 2: Light ({len(light_configs)} configs, "
                f"{actual_workers} workers on {num_gpus} GPU(s)) ──"
            )
            oom = _run_parallel_phase(
                light_configs, max_workers, ctx, report_dirs, errors, gpu_list
            )
            oom_retry.extend(oom)

        # --- Retry: OOM failures, one at a time ------------------------------
        if oom_retry:
            _retry_oom_configs(oom_retry, ctx, report_dirs, errors, gpu_list)

    except KeyboardInterrupt:
        typer.echo("\nBatch run interrupted by user.")
        # Pools are already shut down by the signal handler + finally blocks.
        # Still generate a partial comparison report with what we have.

    total_elapsed = time.monotonic() - t0
    _report_batch_summary(report_dirs, errors, total_elapsed, output_dir)


def _run_parallel_phase(
    configs: list[Path],
    total_workers: int,
    mp_context,
    report_dirs: list[Path],
    errors: list[tuple[str, str]],
    gpu_list: list[int],
) -> list[Path]:
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

    Returns:
        List of config paths that failed with OOM (for retry).
    """
    num_gpus = len(gpu_list)
    workers_per_gpu = max(total_workers // num_gpus, 1)
    total = len(configs)
    oom_configs: list[Path] = []

    # Single-GPU path: no env-var dance needed.
    if num_gpus == 1:
        return _run_parallel_phase_single_gpu(
            configs, total_workers, mp_context, report_dirs, errors
        )

    # Multi-GPU path: one pool per GPU.
    # Build pools first, then distribute configs round-robin.
    pools: list[tuple[concurrent.futures.ProcessPoolExecutor, int]] = []
    for gpu_id in gpu_list:
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers_per_gpu,
            mp_context=mp_context,
        )
        _register_pool(pool)
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
                for f, (p, g) in list(futures.items()):
                    if g == gpu_id and not f.done():
                        errors.append((p.stem, f"Skipped (GPU {gpu_id} pool dead)"))
                        oom_configs.append(p)
                        futures.pop(f, None)
            except Exception as exc:
                msg = str(exc)
                errors.append((cfg_path.stem, msg))
                typer.echo(
                    f"  [{completed}/{total}] FAIL  {cfg_path.stem}: {exc}"
                )
                if _is_oom_error(exc):
                    oom_configs.append(cfg_path)
    finally:
        for pool, _ in pools:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            _unregister_pool(pool)

    return oom_configs


def _run_parallel_phase_single_gpu(
    configs: list[Path],
    workers: int,
    mp_context,
    report_dirs: list[Path],
    errors: list[tuple[str, str]],
) -> list[Path]:
    """Single-GPU fast path — no GPU pinning needed.

    Returns:
        List of config paths that failed with OOM (for retry).
    """
    workers = min(workers, len(configs))
    total = len(configs)
    oom_configs: list[Path] = []

    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
    )
    _register_pool(pool)
    try:
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
                msg = str(exc)
                errors.append((cfg_path.stem, msg))
                typer.echo(f"  [{completed}/{total}] FAIL  {cfg_path.stem}: {exc}")
                if _is_oom_error(exc):
                    oom_configs.append(cfg_path)
    finally:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _unregister_pool(pool)

    return oom_configs


def _retry_oom_configs(
    configs: list[Path],
    mp_context,
    report_dirs: list[Path],
    errors: list[tuple[str, str]],
    gpu_list: list[int],
) -> None:
    """Retry OOM-failed configs sequentially with a dedicated GPU each.

    Each retry runs in a fresh spawned process so the GPU starts clean.
    Successful retries are removed from the error list.
    """
    num_gpus = len(gpu_list)
    typer.echo(f"\n── OOM Retry ({len(configs)} configs, sequential) ──")

    for i, cfg_path in enumerate(configs, 1):
        gpu_id = gpu_list[i % num_gpus]

        # Fresh pool per retry guarantees a clean CUDA context even when
        # the GPU assignment changes between retries.
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=mp_context
        )
        _register_pool(pool)
        try:
            future = pool.submit(_run_one_experiment_gpu, str(cfg_path), gpu_id)
            try:
                result_path = future.result()
                report_dirs.append(Path(result_path))
                typer.echo(
                    f"  [{i}/{len(configs)}] OK  {cfg_path.stem}  "
                    f"(GPU {gpu_id}, retry)"
                )
                _remove_error(errors, cfg_path.stem)
            except Exception as exc:
                typer.echo(
                    f"  [{i}/{len(configs)}] STILL-FAIL  {cfg_path.stem}: {exc}"
                )
        finally:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            _unregister_pool(pool)


def _remove_error(errors: list[tuple[str, str]], name: str) -> None:
    """Remove the first error entry matching *name* in-place."""
    for j, (n, _msg) in enumerate(errors):
        if n == name:
            errors.pop(j)
            return


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
def prewarm(
    config_dir: Path = typer.Argument(..., help="包含 YAML 配置文件的目录"),
) -> None:
    """预下载实验所需全部模型到本地缓存。

    在并行运行实验前使用，避免多进程同时触发下载导致冲突。
    扫描 *config_dir* 中所有 YAML 文件，提取引用的分类器、
    Stable Diffusion pipeline、LPIPS、FID Inception 权重和 CLIP 模型，
    逐一加载后释放。

    \b
    示例:
      python -m src.cli prewarm configs/
      python -m src.cli run-batch configs/ --parallel --gpus 0,1
    """
    if not config_dir.is_dir():
        typer.echo(f"目录不存在: {config_dir}")
        raise typer.Exit(1)

    configs = sorted(config_dir.rglob("*.yaml")) + sorted(config_dir.rglob("*.yml"))
    if not configs:
        typer.echo(f"未找到配置文件: {config_dir}")
        raise typer.Exit(1)

    typer.echo(f"扫描到 {len(configs)} 个配置文件")
    _prewarm_from_configs(configs)


@app.command()
def hello() -> None:
    """测试命令"""
    typer.echo("AIGC Robustness Platform")


if __name__ == "__main__":
    app()
