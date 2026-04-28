import torch


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve a device string to a ``torch.device``.

    When *device_str* is ``"auto"`` the function returns CUDA if any
    GPU is available, otherwise CPU.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
