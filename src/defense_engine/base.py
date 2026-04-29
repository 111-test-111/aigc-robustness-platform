from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch


@dataclass
class DefenseResult:
    """Result of applying a defense."""

    defended: torch.Tensor  # (B, C, H, W) defended samples
    latency_sec: float  # processing time in seconds
    metadata: dict = field(default_factory=dict)


class Defense(ABC):
    """Abstract base class for defense methods."""

    name: str = "base"

    @abstractmethod
    def apply(self, batch: torch.Tensor, config: dict) -> DefenseResult:
        """Apply defense to input batch.

        Args:
            batch: (B, C, H, W) input images in [0, 1]
            config: defense-specific configuration

        Returns:
            DefenseResult with defended samples and latency
        """
        raise NotImplementedError
