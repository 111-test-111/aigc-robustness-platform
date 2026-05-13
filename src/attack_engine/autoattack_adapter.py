"""AutoAttack adapter for strong Lp-bounded robustness baselines."""

from __future__ import annotations

import importlib
import inspect
import time
from typing import Any

import torch

from src.attack_engine.base import Attack, AttackResult


class AutoAttackAdapter(Attack):
    """Wrap the external AutoAttack package behind the platform attack API.

    AutoAttack is an untargeted robustness-evaluation suite.  This adapter
    keeps it separate from the platform's local PGD implementation so paper
    experiments can compare against a widely used strong Lp-bounded baseline
    without changing the runner or reporting code.
    """

    name: str = "autoattack"

    def generate(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        target_model: torch.nn.Module,
        config: dict,
    ) -> AttackResult:
        if config.get("target_class") is not None:
            raise ValueError(
                "AutoAttackAdapter supports untargeted evaluation only; "
                "remove target_class from the attack config."
            )

        autoattack_mod = self._load_autoattack()
        autoattack_cls = autoattack_mod.AutoAttack
        backend_name = getattr(autoattack_mod, "__name__", "autoattack")

        norm: str = str(config.get("norm", "Linf"))
        eps: float = float(config.get("eps", 0.03))
        version: str = str(config.get("version", "standard"))
        batch_size: int = int(config.get("batch_size", config.get("bs", batch.shape[0])))
        seed: int | None = config.get("seed")
        verbose: bool = bool(config.get("verbose", False))
        attacks_to_run = config.get("attacks_to_run")
        log_path = config.get("log_path")

        was_training = target_model.training
        target_params = list(target_model.parameters())
        target_requires_grad = [p.requires_grad for p in target_params]
        target_model.eval()
        for param in target_params:
            param.requires_grad_(False)

        start = time.monotonic()
        try:
            requested_kwargs: dict[str, Any] = {
                "norm": norm,
                "eps": eps,
                "seed": seed,
                "version": version,
                "verbose": verbose,
                "device": str(batch.device),
            }
            if log_path:
                requested_kwargs["log_path"] = str(log_path)

            adversary_kwargs = self._filter_supported_kwargs(autoattack_cls, requested_kwargs)
            adversary = autoattack_cls(target_model, **adversary_kwargs)
            if seed is not None:
                adversary.seed = int(seed)
            if attacks_to_run:
                adversary.attacks_to_run = list(attacks_to_run)

            adv = self._extract_adversarial_tensor(
                self._run_standard_evaluation(adversary, batch, labels, batch_size)
            )
        finally:
            for param, requires_grad in zip(target_params, target_requires_grad):
                param.requires_grad_(requires_grad)
            if was_training:
                target_model.train()

        if batch.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - start

        adv = adv.detach().to(batch.device).clamp(0, 1)
        with torch.no_grad():
            orig_pred = target_model(batch).argmax(dim=1)
            adv_pred = target_model(adv).argmax(dim=1)
            success = adv_pred != orig_pred

        estimated_queries = int(config.get(
            "estimated_queries_per_sample",
            self._estimate_queries(version, attacks_to_run),
        ))

        return AttackResult(
            adversarial=adv,
            success=success,
            queries=[estimated_queries] * batch.shape[0],
            metadata={
                "backend": backend_name,
                "norm": norm,
                "eps": eps,
                "version": version,
                "attacks_to_run": list(attacks_to_run) if attacks_to_run else None,
                "batch_size": batch_size,
                "seed": seed,
                "estimated_queries_per_sample": estimated_queries,
                "queries_are_estimated": True,
                "elapsed_sec": round(elapsed, 4),
            },
        )

    @staticmethod
    def _load_autoattack():
        errors: list[ImportError] = []
        for module_name in ("autoattack", "pyautoattack"):
            try:
                return importlib.import_module(module_name)
            except ImportError as exc:
                errors.append(exc)
        raise ImportError(
            "AutoAttack is not installed. Install the optional baseline "
            "dependencies with `pip install -e '.[baselines]'` or "
            "`pip install pyautoattack>=0.2.0`."
        ) from errors[-1]

    @staticmethod
    def _filter_supported_kwargs(autoattack_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(autoattack_cls)
        except (TypeError, ValueError):
            return kwargs
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    @staticmethod
    def _run_standard_evaluation(adversary: Any, batch: torch.Tensor, labels: torch.Tensor, batch_size: int):
        method = adversary.run_standard_evaluation
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None

        if signature is not None and "batch_size" in signature.parameters:
            return method(batch, labels, batch_size=batch_size)
        if signature is not None and "bs" in signature.parameters:
            return method(batch, labels, bs=batch_size)

        try:
            return method(batch, labels, bs=batch_size)
        except TypeError as exc:
            if "bs" not in str(exc):
                raise
            try:
                return method(batch, labels, batch_size=batch_size)
            except TypeError as batch_size_exc:
                raise ImportError(
                    "Installed AutoAttack package exposes an unsupported "
                    "`run_standard_evaluation` signature."
                ) from batch_size_exc

    @staticmethod
    def _extract_adversarial_tensor(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError(
            "AutoAttack run_standard_evaluation returned an unsupported "
            f"type: {type(output)!r}."
        )

    @staticmethod
    def _estimate_queries(version: str, attacks_to_run: Any) -> int:
        """Return a conservative per-sample query estimate for reporting.

        AutoAttack internally short-circuits attacks once samples are already
        broken and does not expose exact per-sample query counts through the
        public API.  The returned value is therefore marked as estimated in
        metadata and can be overridden in configs.
        """
        if attacks_to_run:
            lookup = {
                "apgd-ce": 100,
                "apgd-dlr": 100,
                "apgd-t": 100,
                "fab": 100,
                "fab-t": 100,
                "square": 5000,
            }
            return sum(lookup.get(str(name), 0) for name in attacks_to_run)

        if version == "plus":
            return 2 * 5 * 100 + 2 * 5 * 100 + 5000 + 9 * 100 + 9 * 100
        if version == "rand":
            return 2 * 20 * 100
        return 100 + 100 + 100 + 5000
