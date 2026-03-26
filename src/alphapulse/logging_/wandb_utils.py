from typing import Any


def init_wandb(
    project: str,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> None:
    import wandb

    wandb.init(project=project, name=name, config=config or {}, **kwargs)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    import wandb

    wandb.log(metrics, step=step)


def log_backtest_results(metrics: dict[str, float], prefix: str = "backtest") -> None:
    import wandb

    prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()}
    wandb.log(prefixed)
