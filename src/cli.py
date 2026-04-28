from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="AIGC 无限制对抗样本攻防验证平台")


@app.command()
def run(
    config: Path = typer.Argument(..., help="实验配置 YAML 路径", exists=True),
) -> None:
    """运行攻防验证实验"""
    from src.task_runner import run_experiment

    typer.echo(f"加载配置: {config}")
    output_dir = run_experiment(config)
    typer.echo(f"实验完成，结果保存至: {output_dir}")


@app.command()
def run_batch(
    config_dir: Path = typer.Argument(..., help="包含 YAML 配置文件的目录"),
) -> None:
    """批量运行目录中的所有实验并生成对比报告"""
    from src.task_runner import run_experiment

    if not config_dir.is_dir():
        typer.echo(f"目录不存在: {config_dir}")
        raise typer.Exit(1)

    configs = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
    if not configs:
        typer.echo(f"未找到配置文件: {config_dir}")
        raise typer.Exit(1)

    typer.echo(f"找到 {len(configs)} 个配置文件")
    output_dirs: list[Path] = []

    for cfg_path in configs:
        typer.echo(f"运行: {cfg_path.name}")
        output_dir = run_experiment(cfg_path)
        output_dirs.append(output_dir)
        typer.echo(f"  完成: {output_dir}")

    # Generate comparison report
    _generate_comparison(output_dirs)
    typer.echo("对比报告已生成")


def _generate_comparison(output_dirs: list[Path]) -> None:
    """Generate comparison charts and report from multiple experiments."""
    from src.reporting.charts import generate_metric_bars, generate_radar

    all_metrics: list[dict[str, float]] = []
    labels: list[str] = []

    for d in output_dirs:
        metrics_path = d / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            all_metrics.append(metrics)
            labels.append(d.name)

    if not all_metrics:
        return

    # Create comparison output directory
    comp_dir = output_dirs[0].parent / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    (comp_dir / "figures").mkdir(exist_ok=True)

    # Generate comparison bar chart
    generate_metric_bars(
        all_metrics,
        labels,
        save_path=comp_dir / "figures" / "comparison_bars.png",
        title="实验对比",
    )

    # Generate comparison radar for first experiment
    # (radar shows one experiment at a time)
    if all_metrics:
        # Normalize metrics for radar (select key metrics)
        radar_keys = [
            k
            for k in all_metrics[0]
            if any(x in k for x in ["_asr", "_robust_accuracy", "_lpips", "_clip_score"])
        ]
        if radar_keys:
            radar_metrics = {k: all_metrics[0][k] for k in radar_keys[:6]}
            generate_radar(
                radar_metrics,
                save_path=comp_dir / "figures" / f"radar_{labels[0]}.png",
                title=f"{labels[0]} 鲁棒性",
            )

    # Save combined metrics
    combined = dict(zip(labels, all_metrics))
    (comp_dir / "comparison.json").write_text(json.dumps(combined, indent=2))


@app.command()
def ui(
    model: str = "resnet50",
    device: str = "cpu",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio web interface."""
    from src.web_ui import create_app

    application = create_app(model, device)
    application.launch(server_port=port, share=share)


@app.command()
def hello() -> None:
    """测试命令"""
    typer.echo("AIGC Robustness Platform")
