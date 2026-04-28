"""PGD (Projected Gradient Descent) attack implementation."""

import torch
import torch.nn as nn

from src.attack_engine.base import Attack, AttackResult


class PGDAttack(Attack):
    """Iterative gradient sign attack (Madry et al., 2018).

    Repeatedly applies small FGSM steps and projects back onto the
    eps-ball around the original input.
    """

    name: str = "pgd"

    def generate(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        target_model: torch.nn.Module,
        config: dict,
    ) -> AttackResult:
        eps: float = config.get("eps", 0.03)
        alpha: float = config.get("alpha", 0.007)
        steps: int = config.get("steps", 20)
        random_start: bool = config.get("random_start", True)

        # Initialize adversarial examples
        if random_start:
            adv = batch + torch.empty_like(batch).uniform_(-eps, eps)
            adv = torch.clamp(adv, 0, 1).detach()
        else:
            adv = batch.clone().detach()

        # Iterative attack loop
        for _ in range(steps):
            adv.requires_grad_(True)
            logits = target_model(adv)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()

            # Gradient ascent step
            adv = adv + alpha * adv.grad.sign()

            # Project back to eps-ball around original
            delta = torch.clamp(adv - batch, -eps, eps)
            adv = torch.clamp(batch + delta, 0, 1).detach()

        # Evaluate attack success
        with torch.no_grad():
            orig_pred = target_model(batch).argmax(dim=1)
            adv_pred = target_model(adv).argmax(dim=1)
            success = orig_pred != adv_pred

        return AttackResult(
            adversarial=adv,
            success=success,
            queries=[steps] * batch.shape[0],
            metadata={
                "eps": eps,
                "alpha": alpha,
                "steps": steps,
                "elapsed_sec": 0.0,
            },
        )
