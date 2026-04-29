import platform

import torch


def _is_supported_apple_silicon_mps() -> bool:
    """Return whether this process can use Apple Silicon GPU acceleration."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False

    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve a device string to a ``torch.device``.

    When *device_str* is ``"auto"``, Apple Silicon Macs use MPS when
    available. Other machines keep the standard CUDA-then-CPU behavior.
    """
    if device_str == "auto":
        if _is_supported_apple_silicon_mps():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
