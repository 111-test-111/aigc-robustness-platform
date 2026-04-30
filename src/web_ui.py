"""Gradio web interface for the AIGC adversarial robustness platform.

Provides four tabs:
1. Interactive Demo -- single-image attack/defense visualization
2. Run Experiments -- execute experiment configs and track progress
3. View Results -- browse completed experiment reports, metrics, and figures
4. Compare Experiments -- side-by-side comparison of multiple experiments
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image

from src.attack_engine.adv_gan import AdvGANAttack
from src.attack_engine.diffusion_attack import DiffusionAttack
from src.attack_engine.fgsm import FGSMAttack
from src.attack_engine.pgd import PGDAttack
from src.defense_engine.bit_depth import BitDepthDefense
from src.defense_engine.blur import GaussianBlurDefense
from src.defense_engine.diffusion_purification import DiffusionPurificationDefense
from src.defense_engine.jpeg import JPEGDefense
from src.model_zoo.classifiers import load_classifier
from src.utils.device import get_device

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
PAPER_CONFIGS_DIR = CONFIGS_DIR / "paper"
ABLATION_CONFIGS_DIR = CONFIGS_DIR / "ablations"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_experiment_configs() -> list[Path]:
    """Return sorted list of paper and ablation YAML configs."""
    configs = sorted(PAPER_CONFIGS_DIR.glob("*.yaml")) + sorted(
        PAPER_CONFIGS_DIR.glob("*.yml")
    )
    configs.extend(sorted(ABLATION_CONFIGS_DIR.rglob("*.yaml")))
    configs.extend(sorted(ABLATION_CONFIGS_DIR.rglob("*.yml")))
    return configs


def _list_completed_experiments() -> list[str]:
    """Return names of completed experiment directories."""
    if not REPORTS_DIR.exists():
        return []
    return sorted(
        d.name for d in REPORTS_DIR.iterdir() if d.is_dir() and (d / "metrics.json").exists()
    )


def _load_experiment_metrics(experiment_name: str) -> dict[str, Any]:
    """Load structured_metrics.json for an experiment."""
    path = REPORTS_DIR / experiment_name / "structured_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _load_flat_metrics(experiment_name: str) -> dict[str, float]:
    """Load metrics.json for an experiment."""
    path = REPORTS_DIR / experiment_name / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _load_report_markdown(experiment_name: str) -> str:
    """Load the generated report.md for an experiment."""
    path = REPORTS_DIR / experiment_name / "report.md"
    if path.exists():
        return path.read_text()
    return f"No report found for {experiment_name}"


def _list_figures(experiment_name: str) -> list[Path]:
    """Return list of figure image paths for an experiment."""
    figures_dir = REPORTS_DIR / experiment_name / "figures"
    if not figures_dir.exists():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in figures_dir.iterdir() if p.suffix.lower() in exts)


def _list_sample_dirs(experiment_name: str) -> list[str]:
    """Return relative paths of sample subdirectories."""
    samples_dir = REPORTS_DIR / experiment_name / "samples"
    if not samples_dir.exists():
        return []
    result: list[str] = []
    for subdir in sorted(samples_dir.iterdir()):
        if subdir.is_dir():
            result.append(str(subdir.relative_to(REPORTS_DIR / experiment_name)))
            # Include nested dirs (e.g. adversarial/fgsm)
            for nested in sorted(subdir.iterdir()):
                if nested.is_dir():
                    result.append(str(nested.relative_to(REPORTS_DIR / experiment_name)))
    return result


def _load_sample_images(experiment_name: str, subdir: str) -> list[Path]:
    """Load sample image paths from a subdirectory."""
    base = REPORTS_DIR / experiment_name
    # Try as absolute path first (includes "samples/" prefix), then relative to samples/
    full_dir = base / subdir
    if not full_dir.exists():
        full_dir = base / "samples" / subdir
    if not full_dir.exists():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in full_dir.iterdir() if p.suffix.lower() in exts)[:16]


def _metrics_to_table(metrics: dict[str, float]) -> list[list[str]]:
    """Convert flat metrics dict to a list of [key, value] rows."""
    rows = []
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            rows.append([k, f"{v:.4f}"])
        else:
            rows.append([k, str(v)])
    return rows


def _structured_to_attack_table(structured: dict) -> list[list[str]]:
    """Convert structured metrics to attack comparison table rows."""
    rows = []
    for name, info in structured.get("attacks", {}).items():
        rows.append([
            name,
            f"{info.get('asr', 0) * 100:.1f}%",
            f"{info.get('asr_on_clean_correct', 0) * 100:.1f}%",
            f"{info.get('clean_accuracy', 0) * 100:.1f}%",
            f"{info.get('adversarial_accuracy', 0) * 100:.1f}%",
            f"{info.get('prediction_change_rate', 0) * 100:.1f}%",
        ])
    return rows


def _structured_to_defense_table(structured: dict) -> list[list[str]]:
    """Convert structured metrics to defense comparison table rows."""
    rows = []
    for name, info in structured.get("defenses", {}).items():
        lat = info.get("latency", {})
        rows.append([
            name,
            f"{info.get('robust_accuracy', 0) * 100:.1f}%",
            f"{info.get('clean_accuracy_drop', 0) * 100:.1f}%",
            f"{info.get('clean_defended_accuracy', 0) * 100:.1f}%",
            f"{lat.get('mean', 0):.4f}s",
        ])
    return rows


# ---------------------------------------------------------------------------
# Tab 1: Interactive Demo
# ---------------------------------------------------------------------------

def _build_interactive_tab(model_name: str, device: str) -> None:
    """Build the interactive single-image demo tab."""
    resolved_device = get_device(device)
    model = load_classifier(model_name, "imagenet", resolved_device)

    attacks = {
        "FGSM": FGSMAttack(),
        "PGD": PGDAttack(),
        "AdvGAN": AdvGANAttack(),
        "Diffusion (mock)": DiffusionAttack(),
    }
    defenses: dict[str, Any] = {
        "None": None,
        "JPEG": JPEGDefense(),
        "Gaussian Blur": GaussianBlurDefense(),
        "Bit Depth": BitDepthDefense(),
        "Diffusion Purification (mock)": DiffusionPurificationDefense(),
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
        noise_level: float,
    ) -> tuple:
        if image is None:
            return (None, None, None, "No image provided", "No image provided", "")

        batch = to_tensor(image).unsqueeze(0).to(resolved_device)
        with torch.no_grad():
            labels = model(batch).argmax(dim=1)

        attack = attacks[attack_name]
        config: dict[str, Any] = {"eps": eps}
        if attack_name == "PGD":
            config["alpha"] = eps / 4
            config["steps"] = int(steps)
        elif attack_name == "AdvGAN":
            config["epochs"] = int(steps)
        elif attack_name == "Diffusion (mock)":
            config = {"backend": "mock", "num_candidates": 3, "strength": 0.7}

        result = attack.generate(batch, labels, model, config)

        defended: torch.Tensor | None = None
        if defense_name != "None":
            defense = defenses[defense_name]
            d_config: dict[str, Any] = {}
            if defense_name == "JPEG":
                d_config["quality"] = int(quality)
            elif defense_name == "Diffusion Purification (mock)":
                d_config = {"backend": "mock", "noise_level": noise_level}
            d_result = defense.apply(result.adversarial, d_config)
            defended = d_result.defended

        orig_img = to_pil(batch[0].detach().cpu().clamp(0, 1))
        adv_img = to_pil(result.adversarial[0].detach().cpu().clamp(0, 1))
        def_img = (
            to_pil(defended[0].detach().cpu().clamp(0, 1))
            if defended is not None
            else None
        )

        delta = (result.adversarial - batch).abs().max().item()
        success = "Yes" if result.success[0].item() else "No"

        with torch.no_grad():
            clean_pred = model(batch).argmax(dim=1).item()
            adv_pred = model(result.adversarial).argmax(dim=1).item()
        pred_change = "Yes" if adv_pred != clean_pred else "No"

        metrics_text = (
            f"L-inf: {delta:.4f}\n"
            f"Attack Success: {success}\n"
            f"Clean prediction: class {clean_pred}\n"
            f"Adversarial prediction: class {adv_pred}\n"
            f"Prediction changed: {pred_change}"
        )

        info = f"Attack: {attack_name}, Defense: {defense_name}"
        return orig_img, adv_img, def_img, metrics_text, info

    with gr.Tab("Interactive Demo"):
        gr.Markdown("## Single-Image Attack & Defense Demo")
        gr.Markdown(
            "Upload an image, select attack and defense methods, and visualize results."
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")
                attack_name = gr.Dropdown(
                    choices=["FGSM", "PGD", "AdvGAN", "Diffusion (mock)"],
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
                    choices=[
                        "None",
                        "JPEG",
                        "Gaussian Blur",
                        "Bit Depth",
                        "Diffusion Purification (mock)",
                    ],
                    value="None",
                    label="Defense Method",
                )
                quality = gr.Slider(
                    minimum=10, maximum=100, value=75, step=5, label="JPEG Quality"
                )
                noise_level = gr.Slider(
                    minimum=0.01,
                    maximum=0.5,
                    value=0.1,
                    step=0.01,
                    label="Diffusion Noise Level",
                )
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column():
                with gr.Row():
                    orig_output = gr.Image(type="pil", label="Original")
                    adv_output = gr.Image(type="pil", label="Adversarial")
                    def_output = gr.Image(type="pil", label="Defended")
                metrics_text = gr.Textbox(label="Metrics", lines=5)
                info_text = gr.Textbox(label="Configuration")

        run_btn.click(
            fn=process,
            inputs=[
                input_image,
                attack_name,
                eps,
                steps,
                defense_name,
                quality,
                noise_level,
            ],
            outputs=[orig_output, adv_output, def_output, metrics_text, info_text],
        )


# ---------------------------------------------------------------------------
# Tab 2: Run Experiments
# ---------------------------------------------------------------------------

def _build_run_experiments_tab() -> None:
    """Build the experiment execution tab."""
    configs = _list_experiment_configs()
    config_names = [c.stem for c in configs]
    config_map = {c.stem: c for c in configs}

    def refresh_configs() -> gr.update:
        nonlocal config_map
        configs_now = _list_experiment_configs()
        names = [c.stem for c in configs_now]
        config_map = {c.stem: c for c in configs_now}
        return gr.update(choices=names, value=names[0] if names else None)

    def run_single(config_name: str) -> str:
        cfg_path = config_map.get(config_name)
        if cfg_path is None:
            return f"Config not found: {config_name}"
        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.cli", "run", str(cfg_path)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=600,
            )
            output = result.stdout + "\n" + result.stderr
            if result.returncode == 0:
                return f"Experiment completed successfully!\n\n{output}"
            return f"Experiment failed (exit code {result.returncode}):\n\n{output}"
        except subprocess.TimeoutExpired:
            return "Experiment timed out after 600 seconds."
        except Exception as e:
            return f"Error running experiment: {e}"

    def run_all() -> str:
        """Run the canonical paper suite sequentially using run_batch."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.cli", "run-batch", str(PAPER_CONFIGS_DIR)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=3600,
            )
            output = result.stdout + "\n" + result.stderr
            if result.returncode == 0:
                return f"All experiments completed!\n\n{output}"
            return f"Batch run failed (exit code {result.returncode}):\n\n{output}"
        except subprocess.TimeoutExpired:
            return "Batch run timed out after 3600 seconds."
        except Exception as e:
            return f"Error running batch: {e}"

    with gr.Tab("Run Experiments"):
        gr.Markdown("## Execute Experiment Configurations")
        gr.Markdown(
            "Select a paper or ablation config to run, or run the paper suite in batch mode."
        )

        with gr.Row():
            with gr.Column():
                config_dropdown = gr.Dropdown(
                    choices=config_names,
                    value=config_names[0] if config_names else None,
                    label="Experiment Config",
                    info="Select from configs/paper/ and configs/ablations/",
                )
                refresh_btn = gr.Button("Refresh Config List")
                with gr.Row():
                    run_single_btn = gr.Button(
                        "Run Selected Experiment", variant="primary"
                    )
                    run_all_btn = gr.Button(
                        "Run Paper Suite", variant="secondary"
                    )

            with gr.Column():
                output_text = gr.Textbox(
                    label="Execution Output",
                    lines=20,
                    max_lines=50,
                )

        refresh_btn.click(fn=refresh_configs, outputs=[config_dropdown])
        run_single_btn.click(fn=run_single, inputs=[config_dropdown], outputs=[output_text])
        run_all_btn.click(fn=run_all, inputs=[], outputs=[output_text])


# ---------------------------------------------------------------------------
# Tab 3: View Results
# ---------------------------------------------------------------------------

def _build_view_results_tab() -> None:
    """Build the experiment results viewing tab."""
    experiment_names = _list_completed_experiments()

    def refresh_experiments() -> gr.update:
        names = _list_completed_experiments()
        return gr.update(choices=names, value=names[0] if names else None)

    def load_experiment(name: str) -> tuple:
        if not name:
            return "", [], [], [], "", ""

        # Report markdown
        report = _load_report_markdown(name)

        # Attack table
        structured = _load_experiment_metrics(name)
        attack_rows = _structured_to_attack_table(structured)
        defense_rows = _structured_to_defense_table(structured)

        # Figures
        figures = _list_figures(name)
        figure_paths = figures if figures else []

        # Sample directories
        sample_dirs = _list_sample_dirs(name)
        samples_info = "\n".join(f"- {d}" for d in sample_dirs) if sample_dirs else "No samples found"

        # Flat metrics
        flat = _load_flat_metrics(name)
        flat_rows = _metrics_to_table(flat)

        return (
            report,
            attack_rows,
            defense_rows,
            figure_paths,
            samples_info,
            flat_rows,
        )

    def load_sample_gallery(name: str, subdir: str) -> list[str]:
        """Load sample images for the gallery."""
        if not name or not subdir:
            return []
        images = _load_sample_images(name, subdir)
        return [str(p) for p in images]

    with gr.Tab("View Results"):
        gr.Markdown("## Experiment Results Viewer")
        gr.Markdown("Browse completed experiments, reports, metrics, and visualizations.")

        with gr.Row():
            experiment_dropdown = gr.Dropdown(
                choices=experiment_names,
                value=experiment_names[0] if experiment_names else None,
                label="Experiment",
            )
            refresh_btn = gr.Button("Refresh")
            load_btn = gr.Button("Load Experiment", variant="primary")

        with gr.Tabs():
            with gr.Tab("Report"):
                report_md = gr.Markdown(label="Experiment Report")

            with gr.Tab("Attack Metrics"):
                attack_table = gr.Dataframe(
                    headers=["Attack", "ASR", "ASR (clean correct)", "Clean Acc", "Adv Acc", "Pred Change"],
                    label="Attack Results",
                    interactive=False,
                )

            with gr.Tab("Defense Metrics"):
                defense_table = gr.Dataframe(
                    headers=["Attack-Defense", "Robust Acc", "Clean Drop", "Clean Defended Acc", "Latency"],
                    label="Defense Results",
                    interactive=False,
                )

            with gr.Tab("All Metrics"):
                all_metrics_table = gr.Dataframe(
                    headers=["Metric", "Value"],
                    label="All Metrics",
                    interactive=False,
                )

            with gr.Tab("Figures"):
                figure_gallery = gr.Gallery(
                    label="Experiment Figures",
                    columns=2,
                    height=400,
                    object_fit="contain",
                )

            with gr.Tab("Samples"):
                samples_info = gr.Textbox(label="Sample Directories", lines=8, interactive=False)
                sample_subdir = gr.Textbox(
                    label="Enter sample subdirectory (e.g. adversarial/fgsm)",
                    placeholder="adversarial/fgsm",
                )
                sample_gallery = gr.Gallery(
                    label="Sample Images",
                    columns=4,
                    height=300,
                    object_fit="contain",
                )
                load_samples_btn = gr.Button("Load Samples")

        refresh_btn.click(fn=refresh_experiments, outputs=[experiment_dropdown])
        load_btn.click(
            fn=load_experiment,
            inputs=[experiment_dropdown],
            outputs=[
                report_md,
                attack_table,
                defense_table,
                figure_gallery,
                samples_info,
                all_metrics_table,
            ],
        )
        load_samples_btn.click(
            fn=load_sample_gallery,
            inputs=[experiment_dropdown, sample_subdir],
            outputs=[sample_gallery],
        )


# ---------------------------------------------------------------------------
# Tab 4: Compare Experiments
# ---------------------------------------------------------------------------

def _build_compare_tab() -> None:
    """Build the experiment comparison tab."""
    experiment_names = _list_completed_experiments()

    def refresh_experiments() -> gr.update:
        names = _list_completed_experiments()
        return gr.update(choices=names)

    def compare(selected: list[str]) -> tuple:
        if not selected or len(selected) < 2:
            return (
                "Select at least 2 experiments to compare.",
                [],
                [],
                [],
            )

        # Build attack comparison table
        attack_headers = ["Experiment", "Attack", "ASR", "ASR (clean correct)", "Clean Acc", "Adv Acc"]
        attack_rows: list[list[str]] = []

        # Build defense comparison table
        defense_headers = ["Experiment", "Attack-Defense", "Robust Acc", "Clean Drop", "Latency"]
        defense_rows: list[list[str]] = []

        # Build summary table
        summary_headers = ["Metric"] + selected
        summary_row_map: dict[str, list[str]] = {}

        for name in selected:
            structured = _load_experiment_metrics(name)
            flat = _load_flat_metrics(name)

            # Attacks
            for atk_name, info in structured.get("attacks", {}).items():
                attack_rows.append([
                    name,
                    atk_name,
                    f"{info.get('asr', 0) * 100:.1f}%",
                    f"{info.get('asr_on_clean_correct', 0) * 100:.1f}%",
                    f"{info.get('clean_accuracy', 0) * 100:.1f}%",
                    f"{info.get('adversarial_accuracy', 0) * 100:.1f}%",
                ])

            # Defenses
            for def_name, info in structured.get("defenses", {}).items():
                lat = info.get("latency", {})
                defense_rows.append([
                    name,
                    def_name,
                    f"{info.get('robust_accuracy', 0) * 100:.1f}%",
                    f"{info.get('clean_accuracy_drop', 0) * 100:.1f}%",
                    f"{lat.get('mean', 0):.4f}s",
                ])

            # Summary metrics
            for key in [
                "fgsm_asr_on_clean_correct",
                "pgd_asr_on_clean_correct",
                "fgsm_clean_accuracy",
                "pgd_clean_accuracy",
            ]:
                if key in flat:
                    if key not in summary_row_map:
                        summary_row_map[key] = [key]
                    # Pad if needed
                    while len(summary_row_map[key]) < len(selected) + 1:
                        summary_row_map[key].append("-")
                    idx = selected.index(name) + 1
                    summary_row_map[key][idx] = f"{flat[key]:.4f}"

        # Fill missing values in summary
        summary_rows = list(summary_row_map.values())
        for row in summary_rows:
            while len(row) < len(selected) + 1:
                row.append("-")

        # Load comparison report if it exists
        comparison_report = ""
        comp_dir = REPORTS_DIR / "comparison"
        comp_file = comp_dir / "comparison.md"
        if comp_file.exists():
            comparison_report = comp_file.read_text()
        elif (REPORTS_DIR / selected[0]).exists():
            # Try to find comparison in parent dir of first experiment
            parent = (REPORTS_DIR / selected[0]).parent / "comparison" / "comparison.md"
            if parent.exists():
                comparison_report = parent.read_text()

        # Comparison figures
        comp_figures: list[str] = []
        if comp_dir.exists():
            fig_dir = comp_dir / "figures"
            if fig_dir.exists():
                comp_figures = [
                    str(p) for p in sorted(fig_dir.iterdir())
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ]

        summary_table = [summary_headers] + summary_rows if summary_rows else []

        return (
            comparison_report,
            attack_rows,
            defense_rows,
            comp_figures,
        )

    with gr.Tab("Compare Experiments"):
        gr.Markdown("## Experiment Comparison")
        gr.Markdown("Select two or more experiments to compare side by side.")

        with gr.Row():
            experiment_checkboxes = gr.CheckboxGroup(
                choices=experiment_names,
                label="Select Experiments to Compare",
            )
            refresh_btn = gr.Button("Refresh")
            compare_btn = gr.Button("Compare", variant="primary")

        with gr.Tabs():
            with gr.Tab("Comparison Report"):
                comparison_md = gr.Markdown(label="Comparison Report")

            with gr.Tab("Attack Comparison"):
                comparison_attack = gr.Dataframe(
                    headers=["Experiment", "Attack", "ASR", "ASR (clean correct)", "Clean Acc", "Adv Acc"],
                    label="Attack Comparison",
                    interactive=False,
                )

            with gr.Tab("Defense Comparison"):
                comparison_defense = gr.Dataframe(
                    headers=["Experiment", "Attack-Defense", "Robust Acc", "Clean Drop", "Latency"],
                    label="Defense Comparison",
                    interactive=False,
                )

            with gr.Tab("Comparison Charts"):
                comparison_figures = gr.Gallery(
                    label="Comparison Charts",
                    columns=2,
                    height=400,
                    object_fit="contain",
                )

        refresh_btn.click(fn=refresh_experiments, outputs=[experiment_checkboxes])
        compare_btn.click(
            fn=compare,
            inputs=[experiment_checkboxes],
            outputs=[
                comparison_md,
                comparison_attack,
                comparison_defense,
                comparison_figures,
            ],
        )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def create_app(model_name: str = "resnet50", device: str = "auto") -> gr.Blocks:
    """Create the Gradio application.

    Args:
        model_name: Name of the classifier to load from the model zoo.
        device: Device string for inference (e.g. "auto", "mps", "cuda", or "cpu").

    Returns:
        A gr.Blocks instance with the full UI.
    """
    with gr.Blocks(title="AIGC Robustness Platform") as app:
        gr.Markdown("# AIGC Adversarial Robustness Platform")
        gr.Markdown(
            "Unified interface for adversarial attack/defense experiments: "
            "run experiments, visualize results, and compare methods."
        )

        with gr.Tabs():
            _build_interactive_tab(model_name, device)
            _build_run_experiments_tab()
            _build_view_results_tab()
            _build_compare_tab()

    return app


def main() -> None:
    """Launch the web UI from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch the AIGC Robustness Platform web UI"
    )
    parser.add_argument("--model", default="resnet50", help="Classifier model name")
    parser.add_argument("--device", default="auto", help="Inference device")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()

    app = create_app(args.model, args.device)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
