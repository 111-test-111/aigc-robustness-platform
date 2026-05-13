"""Inference-only CLIP stub for class-conditional AdvDiff runs.

The ImageNet AdvDiff baseline uses ``ClassEmbedder`` and does not instantiate
CLIP encoders. The upstream LDM module imports ``clip`` at module import time,
so this stub keeps offline server environments importable without pulling the
OpenAI CLIP repository from GitHub.
"""


def load(*args, **kwargs):
    raise RuntimeError("CLIP is not available in the offline AdvDiff runtime.")


def tokenize(*args, **kwargs):
    raise RuntimeError("CLIP is not available in the offline AdvDiff runtime.")
