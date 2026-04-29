import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across all backends.

    Sets the seed for Python's ``random``, NumPy, PyTorch (CPU/MPS and
    all CUDA devices), and configures the cuBLAS workspace to suppress
    non-determinism warnings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
