from __future__ import annotations

import sys
from typing import Any

_WANDB_LOGURU_SINK_ID: int | None = None


def wandb_run_active() -> bool:
    wandb = sys.modules.get("wandb")
    return wandb is not None and getattr(wandb, "run", None) is not None


def attach_wandb_loguru(*, level: str = "INFO") -> None:
    """Send loguru lines to wandb-wrapped stderr for the W&B Logs panel."""
    global _WANDB_LOGURU_SINK_ID
    import sys

    from loguru import logger

    if not wandb_run_active():
        return
    logger.remove()
    _WANDB_LOGURU_SINK_ID = logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
        enqueue=False,
    )


def detach_wandb_loguru() -> None:
    global _WANDB_LOGURU_SINK_ID
    from loguru import logger

    if _WANDB_LOGURU_SINK_ID is not None:
        logger.remove(_WANDB_LOGURU_SINK_ID)
        _WANDB_LOGURU_SINK_ID = None


def log_boosting_round_metrics(
    *,
    model_name: str,
    round_num: int,
    metrics: dict[str, float],
) -> None:
    if not metrics or not wandb_run_active():
        return
    import wandb

    logged: dict[str, Any] = {"train/round": round_num}
    for key, value in metrics.items():
        logged[f"train/{model_name}/{key}"] = value
    wandb.log(logged)


def parse_xgb_evals_log(
    evals_log: dict[str, dict[str, list[float] | list[tuple[float, float]]]],
) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for dataset, metric_map in (evals_log or {}).items():
        for metric_name, values in metric_map.items():
            if not values:
                continue
            last = values[-1]
            parsed[f"{dataset}_{metric_name}"] = float(
                last[0] if isinstance(last, tuple) else last
            )
    return parsed
