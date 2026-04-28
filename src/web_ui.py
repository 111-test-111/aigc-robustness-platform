"""Gradio web interface for the AIGC adversarial robustness platform."""

from __future__ import annotations

from typing import Any

import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image

from src.attack_engine.adv_gan import AdvGANAttack
from src.attack_engine.fgsm import FGSMAttack
from src.attack_engine.pgd import PGDAttack
from src.defense_engine.bit_depth import BitDepthDefense
from src.defense_engine.blur import GaussianBlurDefense
from src.defense_engine.jpeg import JPEGDefense
from src.model_zoo.classifiers import load_classifier


def create_app(model_name: str = "resnet50", device: str = "cpu") -> gr.Blocks:
    """Create the Gradio application.

    Args:
        model_name: Name of the classifier to load from the model zoo.
        device: Device string for inference (e.g. "cpu" or "cuda").

    Returns:
        A gr.Blocks instance with the full UI.
    """
    # Load model once at startup
    model = load_classifier(model_name, "imagenet", torch.device(device))

    # Attack registry
    attacks = {
        "FGSM": FGSMAttack(),
        "PGD": PGDAttack(),
        "AdvGAN": AdvGANAttack(),
    }
    defenses: dict[str, Any] = {
        "None": None,
        "JPEG": JPEGDefense(),
        "Gaussian Blur": GaussianBlurDefense(),
        "Bit Depth": BitDepthDefense(),
    }

    to_tensor = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    to_pil = T.ToPILImage()

    def process(
        image: Image.Image | None,
        attack_name: str,
        eps: float,
        steps: int,
        defense_name: str,
        quality: int,
    ) -> tuple[Image.Image | None, Image.Image | None, Image.Image | None, str, str]:
        """Process an image through attack and optional defense.

        Args:
            image: Uploaded PIL image.
            attack_name: One of FGSM, PGD, AdvGAN.
            eps: Attack perturbation budget.
            steps: Iteration count (steps for PGD, epochs for AdvGAN).
            defense_name: One of None, JPEG, Gaussian Blur, Bit Depth.
            quality: JPEG quality parameter (only used when defense is JPEG).

        Returns:
            Tuple of (original, adversarial, defended, perturbation_info, success_info).
        """
        if image is None:
            return None, None, None, "No image provided", "No image provided"

        # Convert PIL to tensor
        batch = to_tensor(image).unsqueeze(0)
        with torch.no_grad():
            labels = torch.tensor([model(batch).argmax(dim=1).item()])

        # Run attack
        attack = attacks[attack_name]
        config: dict[str, Any] = {"eps": eps}
        if attack_name == "PGD":
            config["steps"] = int(steps)
        elif attack_name == "AdvGAN":
            config["epochs"] = int(steps)

        result = attack.generate(batch, labels, model, config)

        # Run defense if selected
        defended: torch.Tensor | None = None
        if defense_name != "None":
            defense = defenses[defense_name]
            d_config: dict[str, Any] = (
                {"quality": int(quality)} if defense_name == "JPEG" else {}
            )
            d_result = defense.apply(result.adversarial, d_config)
            defended = d_result.defended

        # Convert tensors to PIL images
        orig_img = to_pil(batch[0].clamp(0, 1))
        adv_img = to_pil(result.adversarial[0].clamp(0, 1))
        def_img = to_pil(defended[0].clamp(0, 1)) if defended is not None else None

        # Compute metrics
        delta = (result.adversarial - batch).abs().max().item()
        success = "Yes" if result.success[0].item() else "No"

        return (
            orig_img,
            adv_img,
            def_img,
            f"L-inf: {delta:.4f}",
            f"Attack Success: {success}",
        )

    # Build UI
    with gr.Blocks(title="AIGC Robustness Platform") as app:
        gr.Markdown("# AIGC Adversarial Robustness Platform")
        gr.Markdown(
            "Upload an image, select attack and defense methods, and visualize results."
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")
                attack_name = gr.Dropdown(
                    choices=["FGSM", "PGD", "AdvGAN"],
                    value="FGSM",
                    label="Attack Method",
                )
                eps = gr.Slider(
                    minimum=0.01, maximum=0.5, value=0.03, step=0.01, label="Epsilon"
                )
                steps = gr.Slider(
                    minimum=1, maximum=50, value=20, step=1, label="Steps / Epochs"
                )
                defense_name = gr.Dropdown(
                    choices=["None", "JPEG", "Gaussian Blur", "Bit Depth"],
                    value="None",
                    label="Defense Method",
                )
                quality = gr.Slider(
                    minimum=10, maximum=100, value=75, step=5, label="JPEG Quality"
                )
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column():
                with gr.Row():
                    orig_output = gr.Image(type="pil", label="Original")
                    adv_output = gr.Image(type="pil", label="Adversarial")
                    def_output = gr.Image(type="pil", label="Defended")
                perturbation = gr.Textbox(label="Perturbation")
                success_text = gr.Textbox(label="Attack Success")

        run_btn.click(
            fn=process,
            inputs=[
                input_image,
                attack_name,
                eps,
                steps,
                defense_name,
                quality,
            ],
            outputs=[
                orig_output,
                adv_output,
                def_output,
                perturbation,
                success_text,
            ],
        )

    return app


def main() -> None:
    """Launch the web UI from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch the AIGC Robustness Platform web UI"
    )
    parser.add_argument("--model", default="resnet50", help="Classifier model name")
    parser.add_argument("--device", default="cpu", help="Inference device")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()

    app = create_app(args.model, args.device)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
