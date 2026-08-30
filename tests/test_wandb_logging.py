from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from alphapulse.logging_.wandb_logging import (
    attach_wandb_loguru,
    log_boosting_round_metrics,
    parse_xgb_evals_log,
    wandb_run_active,
)


def test_parse_xgb_evals_log() -> None:
    parsed = parse_xgb_evals_log(
        {
            "train": {"rmse": [0.22, 0.21]},
            "eval": {"rmse": [(0.23, 0.0), (0.22, 0.0)]},
        }
    )
    assert parsed["train_rmse"] == pytest.approx(0.21)
    assert parsed["eval_rmse"] == pytest.approx(0.22)


def test_wandb_run_active_does_not_import_wandb() -> None:
    with patch.dict("sys.modules", {"wandb": None}):
        assert wandb_run_active() is False


def test_log_boosting_round_metrics_logs_to_wandb() -> None:
    mock_wandb = MagicMock()
    with (
        patch("alphapulse.logging_.wandb_logging.wandb_run_active", return_value=True),
        patch.dict("sys.modules", {"wandb": mock_wandb}),
    ):
        log_boosting_round_metrics(
            model_name="XGBoost_0",
            round_num=10,
            metrics={"train_rmse": 0.2, "eval_rmse": 0.21},
        )
    mock_wandb.log.assert_called_once()
    logged = mock_wandb.log.call_args.args[0]
    assert logged["train/round"] == 10
    assert logged["train/XGBoost_0/train_rmse"] == 0.2


def test_attach_wandb_loguru_adds_sink() -> None:
    from alphapulse.logging_.wandb_logging import detach_wandb_loguru

    with patch("alphapulse.logging_.wandb_logging.wandb_run_active", return_value=True):
        attach_wandb_loguru()
        try:
            logger.info("wandb sink smoke test")
        finally:
            detach_wandb_loguru()
            logger.add(lambda msg: None, level="INFO")
