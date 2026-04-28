"""Model registry for registering and discovering model loaders."""

from typing import Callable

MODEL_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator to register a model loader function."""

    def decorator(fn: Callable) -> Callable:
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator
