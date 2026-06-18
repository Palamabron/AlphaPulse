from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from alphapulse.evaluation.shap_report import compute_universal_feature_importance
from alphapulse.evaluation.wandb_diagnostics import log_experiment_diagnostics
from alphapulse.models.xgboost_model import XGBoostModel
from alphapulse.pipeline.model_access import iter_trained_models, model_prediction_map
from alphapulse.pipeline.multi_target import MultiTargetPipeline
from alphapulse.preprocessors.scaling import StandardScalerPreprocessor

FEATURE_COLS = ["f_a", "f_b"]
N = 120


def _multitarget_pipeline() -> tuple[
    MultiTargetPipeline, pd.DataFrame, pd.Series, pd.Series
]:
    rng = np.random.default_rng(3)
    era = pd.Series(np.repeat([f"e{i:03d}" for i in range(12)], N // 12))
    X = pd.DataFrame(
        {
            "f_a": rng.standard_normal(N),
            "f_b": rng.standard_normal(N),
            "era": era.values,
        }
    )
    targets = pd.DataFrame(
        {
            "target": rng.standard_normal(N),
            "target_alpha_20": rng.standard_normal(N),
        }
    )

    def factory() -> XGBoostModel:
        return XGBoostModel(
            params={
                "max_depth": 3,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "objective": "reg:squarederror",
            }
        )

    pipeline = MultiTargetPipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model_factory=factory,
        target_columns=["target", "target_alpha_20"],
        primary_target="target",
    )
    pipeline.fit(X.drop(columns=["era"]), targets, n_rounds=8)
    y_val = targets["target"]
    return pipeline, X, y_val, era


def test_iter_trained_models_multitarget() -> None:
    pipeline, _, _, _ = _multitarget_pipeline()
    models = iter_trained_models(pipeline)
    assert len(models) == 2
    assert all(isinstance(m, XGBoostModel) for m in models)


def test_model_prediction_map_multitarget() -> None:
    pipeline, X, _, _ = _multitarget_pipeline()
    preds = model_prediction_map(pipeline, X, FEATURE_COLS)
    assert set(preds.keys()) == {"target", "target_alpha_20"}
    assert all(len(v) == N for v in preds.values())


def test_compute_universal_feature_importance_multitarget() -> None:
    pipeline, X, _, _ = _multitarget_pipeline()
    importance, label = compute_universal_feature_importance(
        pipeline,
        X.drop(columns=["era"]),
        feature_cols=FEATURE_COLS,
        top_n=10,
    )
    assert len(importance) > 0
    assert "XGBoost" in label


def test_log_experiment_diagnostics_multitarget() -> None:
    pipeline, X, y_val, era_val = _multitarget_pipeline()
    X_feat = X.drop(columns=["era"])
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
            X_val=X_feat,
            y_val=y_val,
            era_val=era_val,
            feature_cols=FEATURE_COLS,
            metrics=metrics,
            log_shap=True,
            log_feature_report=False,
            log_era_importance=False,
            split="validation",
        )

    keys: set[str] = set()
    for call in mock_wandb.log.call_args_list:
        keys.update(call.args[0].keys())
    assert mock_wandb.log.called
    assert "diagnostics/validation/ValidationMmcSharpe" in keys
    assert "diagnostics/validation/ValidationSharpe" in keys
    assert not {k for k in keys if k.endswith(("_table", "_top"))}
