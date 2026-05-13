"""AdvGAN generative adversarial attack implementation."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    @staticmethod
    def _classification_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_class: int | None,
    ) -> tuple[torch.Tensor, str]:
        if target_class is None:
            return -F.cross_entropy(logits, labels), "untargeted_ce_ascent"

        target_labels = torch.full_like(labels, int(target_class))
        return F.cross_entropy(logits, target_labels), "targeted_ce"

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
        target_class: int | None = config.get("target_class", None)

        start = time.monotonic()

        # Create generator and optimizer
        generator = AdvGANGenerator().to(batch.device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

        # Freeze target model parameters while preserving gradients through input.
        was_training = target_model.training
        target_params = list(target_model.parameters())
        target_requires_grad = [p.requires_grad for p in target_params]
        target_model.eval()
        for param in target_params:
            param.requires_grad_(False)

        objective = "untargeted_ce_ascent"

        try:
            # Train generator
            for _ in range(epochs):
                # Generate perturbation
                delta = generator(batch)
                # Clamp perturbation to eps-ball
                delta = torch.clamp(delta, -eps, eps)
                # Create adversarial examples
                adv = torch.clamp(batch + delta, 0, 1)

                logits = target_model(adv)
                loss, objective = self._classification_loss(logits, labels, target_class)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        finally:
            for param, requires_grad in zip(target_params, target_requires_grad):
                param.requires_grad_(requires_grad)
            if was_training:
                target_model.train()

        # Generate final adversarial examples (no grad needed for evaluation)
        with torch.no_grad():
            delta = generator(batch)
            delta = torch.clamp(delta, -eps, eps)
            adv = torch.clamp(batch + delta, 0, 1)

            # Check success
            orig_pred = target_model(batch).argmax(dim=1)
            adv_pred = target_model(adv).argmax(dim=1)
            if target_class is None:
                success = orig_pred != adv_pred
            else:
                success = adv_pred == int(target_class)

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
                "target_class": target_class,
                "objective": objective,
                "elapsed_sec": round(elapsed, 4),
            },
        )


class BatchGeneratorBaselineAttack(AdvGANAttack):
    """Explicit name for the batch-trained small-generator baseline.

    ``AdvGANAttack`` is kept for backward-compatible configs, but new paper
    configs should prefer this name to avoid implying a full offline AdvGAN
    reproduction.
    """

    name: str = "generator_baseline"
