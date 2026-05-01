"""AdvGAN generative adversarial attack implementation."""

import time

import torch
import torch.nn as nn

from src.attack_engine.base import Attack, AttackResult


class AdvGANGenerator(nn.Module):
    """Small CNN generator that maps images to adversarial perturbations.

    Architecture: 4 conv layers with BatchNorm and ReLU, tanh output
    scaled to produce bounded perturbations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdvGANAttack(Attack):
    """AdvGAN adversarial attack (Jia et al., 2019).

    Trains a small generator network on-the-fly for each batch to produce
    adversarial perturbations that fool the target model.
    """

    name: str = "advgan"

    def generate(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        target_model: torch.nn.Module,
        config: dict,
    ) -> AttackResult:
        eps: float = config.get("eps", 0.03)
        epochs: int = config.get("epochs", 50)
        lr: float = config.get("lr", 0.001)

        start = time.monotonic()

        # Create generator and optimizer
        generator = AdvGANGenerator().to(batch.device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

        # Freeze target model during generator training
        target_model.eval()

        # Train generator
        for _ in range(epochs):
            # Generate perturbation
            delta = generator(batch)
            # Clamp perturbation to eps-ball
            delta = torch.clamp(delta, -eps, eps)
            # Create adversarial examples
            adv = torch.clamp(batch + delta, 0, 1)

            # Compute loss: cross-entropy to push toward wrong predictions
            logits = target_model(adv)
            loss = nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Generate final adversarial examples (no grad needed for evaluation)
        with torch.no_grad():
            delta = generator(batch)
            delta = torch.clamp(delta, -eps, eps)
            adv = torch.clamp(batch + delta, 0, 1)

            # Check success
            orig_pred = target_model(batch).argmax(dim=1)
            adv_pred = target_model(adv).argmax(dim=1)
            success = orig_pred != adv_pred

        if batch.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - start

        return AttackResult(
            adversarial=adv,
            success=success,
            queries=[epochs] * batch.shape[0],
            metadata={
                "eps": eps,
                "epochs": epochs,
                "lr": lr,
                "elapsed_sec": round(elapsed, 4),
            },
        )
