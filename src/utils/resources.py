"""Per-experiment resource tracking — GPU memory, GPU utilisation, CPU memory."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class ResourceTracker:
    """Sample GPU and CPU resource usage in a background thread.

    Samples every 0.5 s while the experiment runs; stops on ``stop()``.
    All external dependencies (CUDA, pynvml, psutil) are optional — metrics
    are simply omitted when the underlying library is unavailable.

    Usage::

        tracker = ResourceTracker(device)
        tracker.start()
        # ... run experiment ...
        resource_metrics = tracker.stop()
    """

    def __init__(self, device: Any) -> None:
        self._device = device
        self._device_idx: int = (
            device.index if device is not None and hasattr(device, "index") and device.index is not None else 0
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # Peaks & samples
        self._peak_rss_mb: float = 0.0
        self._gpu_util_pct_samples: list[float] = []

        # Optional imports resolved once
        self._psutil: Any = None
        self._pynvml: Any = None
        self._nvml_handle: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Reset CUDA peak-memory stats and launch background sampler."""
        self._stop.clear()

        # Reset CUDA peak memory tracking so we capture this experiment only.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self._device)
        except Exception:
            pass

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        """Stop sampling and return a dict of resource metrics.

        Keys present depend on which libraries were importable:
        ``gpu_mem_allocated_mb``, ``gpu_mem_reserved_mb``,
        ``cpu_rss_peak_mb``, ``gpu_util_pct_mean``, ``gpu_util_pct_peak``.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

        metrics: dict[str, float] = {}

        # GPU memory -------------------------------------------------------
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.max_memory_allocated(self._device)
                reserved = torch.cuda.max_memory_reserved(self._device)
                metrics["gpu_mem_allocated_mb"] = round(allocated / (1024 * 1024), 1)
                metrics["gpu_mem_reserved_mb"] = round(reserved / (1024 * 1024), 1)
        except Exception:
            pass

        # CPU RSS ----------------------------------------------------------
        if self._peak_rss_mb > 0:
            metrics["cpu_rss_peak_mb"] = round(self._peak_rss_mb, 1)

        # GPU utilisation --------------------------------------------------
        if self._gpu_util_pct_samples:
            metrics["gpu_util_pct_mean"] = round(
                sum(self._gpu_util_pct_samples) / len(self._gpu_util_pct_samples), 1
            )
            metrics["gpu_util_pct_peak"] = round(max(self._gpu_util_pct_samples), 1)

        return metrics

    # ------------------------------------------------------------------
    # Background sampling
    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        """Periodically sample CPU RSS and GPU utilisation."""
        self._init_optional()

        while not self._stop.is_set():
            self._sample_cpu()
            self._sample_gpu_util()
            self._stop.wait(0.5)

        self._teardown_optional()

    def _init_optional(self) -> None:
        """One-time initialisation of optional libraries."""
        try:
            import psutil

            self._psutil = psutil
        except ImportError:
            self._psutil = None

        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_idx)
            self._pynvml = pynvml
        except Exception:
            self._pynvml = None
            self._nvml_handle = None

    def _teardown_optional(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def _sample_cpu(self) -> None:
        if self._psutil is None:
            return
        try:
            rss_mb = self._psutil.Process().memory_info().rss / (1024 * 1024)
            if rss_mb > self._peak_rss_mb:
                self._peak_rss_mb = rss_mb
        except Exception:
            pass

    def _sample_gpu_util(self) -> None:
        if self._pynvml is None or self._nvml_handle is None:
            return
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            self._gpu_util_pct_samples.append(float(util.gpu))
        except Exception:
            pass
