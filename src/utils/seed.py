import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across all backends.

    Sets the seed for Python's ``random``, NumPy, PyTorch (CPU, CUDA,
    and MPS), and configures the cuBLAS workspace for deterministic
    CUDA ops.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    mps = getattr(torch, "mps", None)
    if mps is not None and mps.is_available():
        mps.manual_seed(seed)
