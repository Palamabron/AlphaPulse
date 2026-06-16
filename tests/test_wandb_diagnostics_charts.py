from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from alphapulse.evaluation.wandb_diagnostics import (
    _log_correlation_heatmap,
    _log_feature_exposure,
    _log_horizontal_bar_chart,
    log_experiment_diagnostics,
)
from alphapulse.models.xgboost_model import XGBoostModel
from alphapulse.pipeline.pipeline import Pipeline
from alphapulse.preprocessors.scaling import StandardScalerPreprocessor

FEATURE_COLS = ["f_a", "f_b"]
N = 120


def _pipeline_and_data() -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(1)
    era = pd.Series(np.repeat([f"e{i:03d}" for i in range(12)], N // 12))
    X = pd.DataFrame(
        {
            "f_a": rng.standard_normal(N),
            "f_b": rng.standard_normal(N),
        }
    )
    y = pd.Series(rng.standard_normal(N))
    pipe = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        models=[
            XGBoostModel(
                params={
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                }
            )
        ],
    )
    pipe.fit(X, y, n_rounds=8)
    return pipe, X, y, era


def _collect_logged_keys(mock_wandb: MagicMock) -> set[str]:
    keys: set[str] = set()
    for call in mock_wandb.log.call_args_list:
        payload = call.args[0]
        keys.update(payload.keys())
    return keys


def test_diagnostics_logs_charts_not_raw_tables() -> None:
    pipeline, X, y, era = _pipeline_and_data()
    metrics = {
        "corr_sharpe": 1.0,
        "mmc": 0.01,
        "mmc_sharpe": 0.5,
        "payout_score": 1.2,
        "mean_per_era_correlation": 0.02,
        "std_per_era_correlation": 0.01,
        "max_drawdown": 0.05,
        "pct_positive_eras": 0.8,
        "n_valid_eras": 10,
    }

    mock_wandb = MagicMock()
    mock_wandb.run = MagicMock()
    mock_wandb.Table = MagicMock(side_effect=lambda columns: MagicMock(columns=columns))
    mock_wandb.plot = MagicMock()
    mock_wandb.Image = MagicMock()

    with (
        patch(
            "alphapulse.evaluation.wandb_diagnostics._wandb_active",
            return_value=True,
        ),
        patch.dict("sys.modules", {"wandb": mock_wandb}),
    ):
        log_experiment_diagnostics(
            pipeline=pipeline,
            X_val=X,
            y_val=y,
            era_val=era,
            feature_cols=FEATURE_COLS,
            metrics=metrics,
            log_shap=True,
            log_feature_report=False,
            log_era_importance=False,
        )

    keys = _collect_logged_keys(mock_wandb)
    table_keys = {k for k in keys if k.endswith(("_table", "_top"))}
    assert not table_keys
    assert "diagnostics/mmc" in keys
    assert "diagnostics/mmc_sharpe" in keys
    assert any(k.endswith("_bar") or "correlation" in k for k in keys)


def test_horizontal_bar_chart_logs_image() -> None:
    mock_wandb = MagicMock()
    mock_wandb.Image = MagicMock(return_value="img")
    _log_horizontal_bar_chart(
        mock_wandb,
        labels=["a", "b"],
        values=[0.1, 0.2],
        key="diagnostics/test_bar",
        title="Test",
        xlabel="value",
    )
    mock_wandb.log.assert_called_once()
    assert mock_wandb.log.call_args.args[0]["diagnostics/test_bar"] == "img"


def test_correlation_heatmap_logs_image() -> None:
    mock_wandb = MagicMock()
    mock_wandb.Image = MagicMock(return_value="img")
    corr = {
        "m1": {"m1": 1.0, "m2": 0.3},
        "m2": {"m1": 0.3, "m2": 1.0},
    }
    _log_correlation_heatmap(
        mock_wandb,
        ["m1", "m2"],
        corr,
        "diagnostics/test_heatmap",
        title="Heatmap",
    )
    mock_wandb.log.assert_called_once()


def test_feature_exposure_uses_bar_chart_only() -> None:
    rng = np.random.default_rng(0)
    preds = rng.standard_normal(60)
    features = pd.DataFrame(
        {"f_a": rng.standard_normal(60), "f_b": rng.standard_normal(60)}
    )
    eras = pd.Series(np.repeat([f"e{i:03d}" for i in range(6)], 10))

    mock_wandb = MagicMock()
    mock_wandb.Image = MagicMock(return_value="img")
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        _log_feature_exposure(preds, features, eras)
    keys = _collect_logged_keys(mock_wandb)
    assert "diagnostics/feature_exposure_bar" in keys
    assert "diagnostics/feature_exposure_top" not in keys
