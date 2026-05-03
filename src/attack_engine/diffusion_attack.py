"""Diffusion-based adversarial attack using Stable Diffusion img2img pipeline.

Candidate count (``num_candidates``) defaults to 5, chosen as the knee point
from the {1, 3, 5, 10} ablation sweep
(``configs/ablations/attack/diffusion/``).  At N=5 the marginal attack-success
gain per additional candidate flattens while query cost grows linearly,
maximising ASR per unit query budget.  Paper experiments should confirm the
knee still holds for the target model and dataset under test.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage

from src.attack_engine.base import Attack, AttackResult


class DiffusionAttack(Attack):
    """Generate adversarial samples via Stable Diffusion img2img transformation.

    Pipeline
    --------
    1. For each of *N* rounds, generate one candidate per image via SD img2img
       (or mock transforms for testing).
    2. Evaluate the new candidate batch against the target classifier.
    3. **Early-stop** once every sample in the batch has at least one
       successful candidate — avoiding wasted SD generation and classifier
       inference.
    4. Among each sample's successful candidates, select the one with the
       **highest LPIPS distance** from the original (most perceptually
       diverse perturbation, hardest to defend).
    5. If any sample fails all *N* rounds, apply an optional **fallback**
       pass with increased ``strength`` to attempt recovery.

    Supports two backends:
    - ``sd``: Real Stable Diffusion pipeline (requires diffusers + model
      weights)
    - ``mock``: Simulated diffusion using controllable image transforms (for
      pipeline testing without a GPU)
    """

    name: str = "diffusion"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        target_model: torch.nn.Module,
        config: dict,
    ) -> AttackResult:
        # --- Config -----------------------------------------------------------
        backend: str = config.get("backend", "sd")
        model_id: str = config.get(
            "generator", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        )
        prompt: str = config.get("prompt", "a high quality photo")
        strength: float = config.get("strength", 0.7)
        guidance_scale: float = config.get("guidance_scale", 7.5)
        num_candidates: int = config.get("num_candidates", 5)
        target_class: int | None = config.get("target_class", None)

        # Fallback controls
        fallback_enabled: bool = config.get("fallback_enabled", True)
        fallback_strength_delta: float = config.get("fallback_strength_delta", 0.20)
        fallback_candidates: int = config.get("fallback_candidates", 3)

        B = batch.shape[0]
        device = batch.device

        # Pre-compute original predictions for untargeted success checks.
        with torch.no_grad():
            orig_pred = target_model(batch).argmax(dim=1)

        # Per-sample state
        best_adv = batch.clone()
        best_lpips = torch.full((B,), -1.0, device=device)
        best_success = torch.zeros(B, dtype=torch.bool, device=device)
        queries = [0] * B

        start = time.perf_counter()

        # Load pipeline once (SD backend only).
        pipe = None
        if backend != "mock":
            from src.model_zoo.generators import load_sd_pipeline

            pipe = load_sd_pipeline(model_id, device)

        # --- Candidate loop with early stopping --------------------------------
        for _round_idx in range(num_candidates):
            # Generate one candidate per sample for this round.
            with torch.no_grad():
                if backend == "mock":
                    cand = self._generate_one_mock_round(batch, strength)
                else:
                    cand = self._generate_one_sd_round(
                        pipe, batch, prompt, strength, guidance_scale
                    )

            # Evaluate against target classifier.
            with torch.no_grad():
                cand_pred = target_model(cand).argmax(dim=1)

            # Compute LPIPS for all samples (batched, cheap).  Full-batch
            # is intentional: a sample that succeeded in an earlier round
            # can be overtaken by a higher-LPIPS candidate from a later
            # round (generated because *other* samples still need a win).
            cand_lpips = self._compute_per_sample_lpips(batch, cand)

            for s in range(B):
                queries[s] += 1

                is_success: bool = (
                    cand_pred[s] == target_class
                    if target_class is not None
                    else cand_pred[s] != orig_pred[s]
                )

                if is_success:
                    if cand_lpips[s] > best_lpips[s]:
                        best_lpips[s] = cand_lpips[s]
                        best_adv[s] = cand[s].clone()
                    best_success[s] = True

            # Early-stop: every sample has at least one successful candidate.
            if best_success.all():
                break

        # --- Fallback for samples that failed all N rounds --------------------
        fallback_used = False
        if fallback_enabled and not best_success.all():
            best_adv, best_success, queries, fallback_used = self._apply_fallback(
                batch=batch,
                best_adv=best_adv,
                best_success=best_success,
                best_lpips=best_lpips,
                queries=queries,
                pipe=pipe,
                target_model=target_model,
                orig_pred=orig_pred,
                target_class=target_class,
                backend=backend,
                prompt=prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                fallback_strength_delta=fallback_strength_delta,
                fallback_candidates=fallback_candidates,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
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
                "fallback_used": fallback_used,
                "elapsed_sec": elapsed,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers — single-round generation
    # ------------------------------------------------------------------

    def _generate_one_sd_round(
        self,
        pipe: Any,
        batch: torch.Tensor,
        prompt: str,
        strength: float,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run img2img once, producing one candidate per image.

        Returns:
            Tensor of shape ``(B, C, H, W)`` in ``[0, 1]``.
        """
        to_pil = ToPILImage()
        candidates: list[torch.Tensor] = []

        for i in range(batch.shape[0]):
            pil_img = to_pil(batch[i].clamp(0, 1).cpu())
            result = pipe(
                prompt=prompt,
                image=pil_img,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=20,
            )
            generated = TF.to_tensor(result.images[0]).float()
            candidates.append(generated)

        return torch.stack(candidates).to(batch.device)

    def _generate_one_mock_round(
        self,
        batch: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """Simulate one diffusion round with controllable transforms.

        Returns:
            Tensor of shape ``(B, C, H, W)`` in ``[0, 1]``.
        """
        jitter = T.ColorJitter(
            brightness=0.3 * strength,
            contrast=0.3 * strength,
            saturation=0.2 * strength,
        )
        blur = T.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 0.5) if strength < 0.5 else (0.5, 1.5),
        )

        candidates: list[torch.Tensor] = []
        for i in range(batch.shape[0]):
            img = batch[i].clamp(0, 1).cpu()
            noise = torch.randn_like(img) * strength * 0.1
            noisy_img = torch.clamp(img + noise, 0, 1)
            pil_img = TF.to_pil_image(noisy_img)
            transformed = blur(jitter(pil_img))
            candidates.append(TF.to_tensor(transformed).float())

        return torch.stack(candidates).to(batch.device)

    # ------------------------------------------------------------------
    # Internal helpers — LPIPS
    # ------------------------------------------------------------------

    def _compute_per_sample_lpips(
        self,
        original: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample LPIPS distances.

        Args:
            original: ``(B, C, H, W)`` in ``[0, 1]``.
            candidates: ``(B, C, H, W)`` in ``[0, 1]``.

        Returns:
            ``(B,)`` tensor of LPIPS distances (0 = identical, higher =
            more perceptually different).
        """
        from src.evaluation.quality_metrics import _get_lpips

        orig_scaled = original * 2 - 1
        cand_scaled = candidates * 2 - 1

        fn = _get_lpips(original.device)
        with torch.no_grad():
            distances = fn(orig_scaled, cand_scaled)
        return distances.view(-1)

    # ------------------------------------------------------------------
    # Internal helpers — fallback
    # ------------------------------------------------------------------

    def _apply_fallback(
        self,
        *,
        batch: torch.Tensor,
        best_adv: torch.Tensor,
        best_success: torch.Tensor,
        best_lpips: torch.Tensor,
        queries: list[int],
        pipe: Any,
        target_model: torch.nn.Module,
        orig_pred: torch.Tensor,
        target_class: int | None,
        backend: str,
        prompt: str,
        strength: float,
        guidance_scale: float,
        fallback_strength_delta: float,
        fallback_candidates: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], bool]:
        """Retry failed samples with increased strength.

        Only generates candidates for the sub-batch of samples that failed
        all rounds of the main loop.

        Returns:
            ``(best_adv, best_success, queries, fallback_used)`` — the
            input tensors with element-wise updates where fallback
            candidates succeeded.
        """
        failed_mask = ~best_success
        failed_indices = failed_mask.nonzero(as_tuple=True)[0]
        F = failed_indices.shape[0]
        if F == 0:
            return best_adv, best_success, queries, False

        failed_batch = batch[failed_indices]  # (F, C, H, W)
        fallback_strength = min(strength + fallback_strength_delta, 1.0)

        for _ in range(fallback_candidates):
            with torch.no_grad():
                if backend == "mock":
                    cand = self._generate_one_mock_round(
                        failed_batch, fallback_strength
                    )
                else:
                    cand = self._generate_one_sd_round(
                        pipe, failed_batch, prompt,
                        fallback_strength, guidance_scale,
                    )

            with torch.no_grad():
                cand_pred = target_model(cand).argmax(dim=1)

            # Compute LPIPS for all failed samples in this fallback round.
            cand_lpips = self._compute_per_sample_lpips(failed_batch, cand)

            for local_idx, global_idx in enumerate(failed_indices):
                s = int(global_idx)
                queries[s] += 1

                is_success: bool = (
                    cand_pred[local_idx] == target_class
                    if target_class is not None
                    else cand_pred[local_idx] != orig_pred[s]
                )

                if is_success:
                    if cand_lpips[local_idx] > best_lpips[s]:
                        best_lpips[s] = cand_lpips[local_idx]
                        best_adv[s] = cand[local_idx].clone()
                    best_success[s] = True

            # Short-circuit: all failed samples rescued.
            if best_success.all():
                break

        return best_adv, best_success, queries, True
