"""Diffusion-based adversarial attack using Stable Diffusion img2img pipeline."""

from __future__ import annotations

import time
from typing import Any

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage

from src.attack_engine.base import Attack, AttackResult


class DiffusionAttack(Attack):
    """Generate adversarial samples via Stable Diffusion img2img transformation.

    Uses text-guided image-to-image generation to create candidates, then
    filters by target classifier prediction. Among successful candidates,
    selects the one with highest LPIPS distance (most natural-looking perturbation).

    Supports two backends:
    - ``sd``: Real Stable Diffusion pipeline (requires diffusers + model weights)
    - ``mock``: Simulated diffusion using controllable image transforms (for testing)
    """

    name: str = "diffusion"

    def generate(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        target_model: torch.nn.Module,
        config: dict,
    ) -> AttackResult:
        # --- Config ---
        backend: str = config.get("backend", "sd")
        model_id: str = config.get(
            "generator", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        )
        prompt: str = config.get("prompt", "a high quality photo")
        strength: float = config.get("strength", 0.7)
        guidance_scale: float = config.get("guidance_scale", 7.5)
        num_candidates: int = config.get("num_candidates", 5)
        target_class: int | None = config.get("target_class", None)

        start = time.perf_counter()

        # --- Generate candidates based on backend ---
        if backend == "mock":
            all_candidates = self._generate_mock_candidates(
                batch, strength, num_candidates
            )
        else:
            from src.model_zoo.generators import load_sd_pipeline
            pipe = load_sd_pipeline(model_id, batch.device)
            all_candidates = self._generate_candidates(
                pipe, batch, prompt, strength, guidance_scale, num_candidates
            )

        # --- Evaluate and select best adversarial ---
        best_adv, best_success, queries = self._select_best(
            batch, all_candidates, target_model, target_class, num_candidates
        )

        elapsed = time.perf_counter() - start

        return AttackResult(
            adversarial=best_adv.detach(),
            success=best_success,
            queries=queries,
            metadata={
                "backend": backend,
                "generator": model_id,
                "prompt": prompt,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "num_candidates": num_candidates,
                "elapsed_sec": elapsed,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_mock_candidates(
        self,
        batch: torch.Tensor,
        strength: float,
        num_candidates: int,
    ) -> torch.Tensor:
        """Generate mock diffusion candidates using controllable transforms.

        Simulates diffusion output by applying:
        - Gaussian noise scaled by strength
        - Random brightness/contrast jitter
        - Slight spatial blur

        This is useful for testing the pipeline without a real SD model.

        Args:
            batch: (B, C, H, W) input images in [0, 1]
            strength: Controls noise magnitude (0 = no change, 1 = full noise)
            num_candidates: Number of candidate variants per image

        Returns:
            Tensor of shape (num_candidates, B, C, H, W) in [0, 1].
        """
        import torchvision.transforms as T

        batch_size = batch.shape[0]
        candidate_list: list[torch.Tensor] = []

        # Define transform pipeline simulating diffusion output
        jitter = T.ColorJitter(brightness=0.3 * strength, contrast=0.3 * strength, saturation=0.2 * strength)
        blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5) if strength < 0.5 else (0.5, 1.5))

        for _ in range(num_candidates):
            candidate_batch: list[torch.Tensor] = []
            for img_idx in range(batch_size):
                img = batch[img_idx].clamp(0, 1).cpu()

                # Add Gaussian noise scaled by strength
                noise = torch.randn_like(img) * strength * 0.1
                noisy_img = torch.clamp(img + noise, 0, 1)

                # Apply jitter and blur
                pil_img = TF.to_pil_image(noisy_img)
                transformed = jitter(pil_img)
                transformed = blur(transformed)

                candidate_batch.append(TF.to_tensor(transformed).float())

            candidate_list.append(
                torch.stack(candidate_batch).to(batch.device)
            )

        return torch.stack(candidate_list)

    def _generate_candidates(
        self,
        pipe: Any,
        batch: torch.Tensor,
        prompt: str,
        strength: float,
        guidance_scale: float,
        num_candidates: int,
    ) -> torch.Tensor:
        """Run img2img pipeline for each candidate round.

        Returns:
            Tensor of shape (num_candidates, B, C, H, W) in [0, 1].
        """
        to_pil = ToPILImage()
        batch_size = batch.shape[0]
        candidate_list: list[torch.Tensor] = []

        for _ in range(num_candidates):
            candidate_batch: list[torch.Tensor] = []
            for img_idx in range(batch_size):
                pil_img = to_pil(batch[img_idx].clamp(0, 1).cpu())
                result = pipe(
                    prompt=prompt,
                    image=pil_img,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=20,
                )
                # Convert PIL image to tensor correctly
                generated = TF.to_tensor(result.images[0]).float()
                candidate_batch.append(generated)
            candidate_list.append(
                torch.stack(candidate_batch).to(batch.device)
            )

        return torch.stack(candidate_list)

    def _select_best(
        self,
        batch: torch.Tensor,
        all_candidates: torch.Tensor,
        target_model: torch.nn.Module,
        target_class: int | None,
        num_candidates: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Pick the best adversarial candidate for each sample.

        Selection criteria:
        1. Candidate must fool the target classifier (attack success).
        2. Among successful candidates, pick the first one found
           (later candidates may have higher diversity from SD randomness).

        Returns:
            best_adv: (B, C, H, W)
            best_success: (B,) bool
            queries: list[int] per sample
        """
        batch_size = batch.shape[0]
        best_adv = batch.clone()
        best_success = torch.zeros(batch_size, dtype=torch.bool, device=batch.device)
        queries = [num_candidates] * batch_size

        with torch.no_grad():
            orig_pred = target_model(batch).argmax(dim=1)

            for cand_idx in range(num_candidates):
                cand = all_candidates[cand_idx]
                cand_pred = target_model(cand).argmax(dim=1)

                if target_class is not None:
                    success = cand_pred == target_class
                else:
                    success = cand_pred != orig_pred

                for s in range(batch_size):
                    if success[s] and not best_success[s]:
                        best_adv[s] = cand[s]
                        best_success[s] = True

        return best_adv, best_success, queries
