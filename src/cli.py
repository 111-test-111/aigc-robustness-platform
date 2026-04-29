from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(help="AIGC 无限制对抗样本攻防验证平台")
logger = logging.getLogger(__name__)


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
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="对比报告输出目录"
    ),
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
    report_dirs: list[Path] = []

    for cfg_path in configs:
        typer.echo(f"运行: {cfg_path.name}")
        report_dir = run_experiment(cfg_path)
        report_dirs.append(report_dir)
        typer.echo(f"  完成: {report_dir}")

    # Generate comparison report
    comp_dir = output_dir or (report_dirs[0].parent / "comparison")
    _generate_comparison(report_dirs, comp_dir)
    typer.echo(f"对比报告已生成: {comp_dir}")


def _generate_comparison(report_dirs: list[Path], comp_dir: Path) -> None:
    """Generate comparison charts and report from multiple experiments."""
    from src.reporting.charts import generate_metric_bars, generate_radar
    from src.task_runner import _build_radar_metrics

    all_metrics: list[dict[str, float]] = []
    labels: list[str] = []
    all_structured: list[dict] = []

    for d in report_dirs:
        metrics_path = d / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            # Filter out non-numeric values
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            all_metrics.append(numeric_metrics)
            labels.append(d.name)

        structured_path = d / "structured_metrics.json"
        if structured_path.exists():
            all_structured.append(json.loads(structured_path.read_text()))

    if not all_metrics:
        return

    comp_dir.mkdir(parents=True, exist_ok=True)
    (comp_dir / "figures").mkdir(exist_ok=True)

    # Generate comparison bar chart
    if all_metrics:
        generate_metric_bars(
            all_metrics,
            labels,
            save_path=comp_dir / "figures" / "comparison_bars.png",
            title="实验对比",
        )

    # Generate radar charts for each experiment
    for i, (metrics, label) in enumerate(zip(all_metrics, labels)):
        radar_metrics = _build_radar_metrics(metrics)
        if radar_metrics:
            generate_radar(
                radar_metrics,
                save_path=comp_dir / "figures" / f"radar_{label}.png",
                title=f"{label} 鲁棒性",
            )

    # Generate comparison markdown report
    _generate_comparison_report(all_metrics, all_structured, labels, comp_dir)

    # Save combined metrics
    combined = dict(zip(labels, all_metrics))
    (comp_dir / "comparison.json").write_text(json.dumps(combined, indent=2))

    # Save comparison CSV
    _save_comparison_csv(all_metrics, labels, comp_dir)


def _generate_comparison_report(
    all_metrics: list[dict[str, float]],
    all_structured: list[dict],
    labels: list[str],
    comp_dir: Path,
) -> None:
    """Generate a markdown comparison report."""
    from datetime import datetime, timezone

    lines = [
        "# 实验对比报告",
        "",
        f"> 自动生成于 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## 1. 实验列表",
        "",
        "| 序号 | 实验名称 |",
        "|------|----------|",
    ]

    for i, label in enumerate(labels, 1):
        lines.append(f"| {i} | {label} |")

    lines.extend(["", "## 2. 攻击方法对比", ""])

    # Build attack comparison table
    attack_names: set[str] = set()
    for structured in all_structured:
        attack_names.update(structured.get("attacks", {}).keys())

    if attack_names:
        lines.append("| 实验 | 攻击方法 | ASR | 干净准确率 | 对抗准确率 |")
        lines.append("|------|----------|-----|-----------|-----------|")
        for structured, label in zip(all_structured, labels):
            for attack_name, info in structured.get("attacks", {}).items():
                asr = f"{info.get('asr', 0) * 100:.1f}%"
                clean_acc = f"{info.get('clean_accuracy', 0) * 100:.1f}%"
                adv_acc = f"{info.get('adversarial_accuracy', 0) * 100:.1f}%"
                lines.append(f"| {label} | {attack_name} | {asr} | {clean_acc} | {adv_acc} |")

    lines.extend(["", "## 3. 防御方法对比", ""])

    # Build defense comparison table
    defense_names: set[str] = set()
    for structured in all_structured:
        defense_names.update(structured.get("defenses", {}).keys())

    if defense_names:
        lines.append("| 实验 | 攻击-防御 | 鲁棒准确率 | 干净下降 | 延迟 (s) |")
        lines.append("|------|-----------|-----------|---------|----------|")
        for structured, label in zip(all_structured, labels):
            for defense_name, info in structured.get("defenses", {}).items():
                ra = f"{info.get('robust_accuracy', 0) * 100:.1f}%"
                cad = f"{info.get('clean_accuracy_drop', 0) * 100:.1f}%"
                lat = f"{info.get('latency', {}).get('mean', 0):.4f}"
                lines.append(f"| {label} | {defense_name} | {ra} | {cad} | {lat} |")

    lines.extend(["", "## 4. 可视化", ""])
    lines.append("### 实验对比柱状图")
    lines.append("")
    lines.append("![对比柱状图](figures/comparison_bars.png)")
    lines.append("")

    for label in labels:
        radar_path = f"figures/radar_{label}.png"
        lines.append(f"### {label} 雷达图")
        lines.append("")
        lines.append(f"![{label} 雷达图]({radar_path})")
        lines.append("")

    lines.extend([
        "---",
        "*报告由 AIGC 鲁棒性平台自动生成*",
    ])

    (comp_dir / "comparison.md").write_text("\n".join(lines))


def _save_comparison_csv(
    all_metrics: list[dict[str, float]],
    labels: list[str],
    comp_dir: Path,
) -> None:
    """Save comparison metrics as a wide CSV table."""
    import csv

    # Collect all metric keys
    all_keys: list[str] = []
    seen: set[str] = set()
    for metrics in all_metrics:
        for k in metrics:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    csv_path = comp_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment"] + all_keys)
        for metrics, label in zip(all_metrics, labels):
            row = [label] + [metrics.get(k, "") for k in all_keys]
            writer.writerow(row)


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


if __name__ == "__main__":
    app()
