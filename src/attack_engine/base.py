from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch


@dataclass
class AttackResult:
    """Result of an attack generation."""

    adversarial: torch.Tensor       # (B, C, H, W) adversarial samples
    success: torch.Tensor            # (B,) bool mask indicating attack success
    queries: list[int]               # query count per sample
    metadata: dict = field(default_factory=dict)  # runtime info (elapsed_sec, etc.)


class Attack(ABC):
    """Abstract base class for adversarial attacks."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        target_model: torch.nn.Module,
        config: dict,
    ) -> AttackResult:
        """Generate adversarial samples.

        Args:
            batch: (B, C, H, W) input images in [0, 1]
            labels: (B,) ground truth labels
            target_model: classifier to attack
            config: attack-specific configuration

        Returns:
            AttackResult with adversarial samples and metadata
        """
        raise NotImplementedError
