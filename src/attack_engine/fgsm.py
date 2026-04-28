"""FGSM (Fast Gradient Sign Method) attack implementation."""

import torch
import torch.nn as nn

from src.attack_engine.base import Attack, AttackResult


class FGSMAttack(Attack):
    """Single-step gradient sign attack (Goodfellow et al., 2015)."""

    name: str = "fgsm"

    def generate(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        target_model: torch.nn.Module,
        config: dict,
    ) -> AttackResult:
        eps: float = config.get("eps", 0.03)

        # Enable gradients on input
        x = batch.clone().detach().requires_grad_(True)

        # Forward pass
        logits = target_model(x)
        loss = nn.functional.cross_entropy(logits, labels)

        # Backward pass
        loss.backward()

        # Generate adversarial examples via gradient sign
        grad = x.grad.data
        adv = torch.clamp(x + eps * grad.sign(), 0, 1).detach()

        # Check success (prediction changed)
        with torch.no_grad():
            orig_pred = target_model(batch).argmax(dim=1)
            adv_pred = target_model(adv).argmax(dim=1)
            success = orig_pred != adv_pred

        return AttackResult(
            adversarial=adv,
            success=success,
            queries=[1] * batch.shape[0],
            metadata={"eps": eps, "elapsed_sec": 0.0},
        )
