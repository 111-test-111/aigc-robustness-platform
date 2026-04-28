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
def hello() -> None:
    """测试命令"""
    typer.echo("AIGC Robustness Platform")
