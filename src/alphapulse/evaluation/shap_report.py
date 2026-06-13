from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..models.base import _numeric
from ..models.era_ensemble_model import EraEnsembleModel
from ..models.xgboost_model import XGBoostModel
from ..pipeline.multihead import MultiHeadPipeline
from ..pipeline.pipeline import Pipeline

SHAP_SAMPLE_ROWS = 2000


def _wandb_active() -> bool:
    try:
        import wandb

        return wandb.run is not None
    except ImportError:
        return False


def _xgb_contribs(model: XGBoostModel, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feat = _numeric(X)
    dmat = xgb.DMatrix(feat)
    contribs = model.model.predict(dmat, pred_contribs=True)
    cols = list(feat.columns)
    return np.asarray(contribs, dtype=np.float64), cols


def _collect_xgboost_models(
    pipeline: Pipeline | MultiHeadPipeline,
) -> list[tuple[str, XGBoostModel]]:
    models: list[tuple[str, XGBoostModel]] = []
    if isinstance(pipeline, MultiHeadPipeline):
        for head in pipeline.heads:
            m = head.model
            if isinstance(m, XGBoostModel):
                models.append((m.name, m))
            elif isinstance(m, EraEnsembleModel):
                for sub in m._sub_models:
                    if isinstance(sub, XGBoostModel):
                        models.append((sub.name, sub))
        return models

    for model in pipeline.models:
        if isinstance(model, XGBoostModel):
            models.append((model.name, model))
        elif isinstance(model, EraEnsembleModel):
            for sub in model._sub_models:
                if isinstance(sub, XGBoostModel):
                    models.append((sub.name, sub))
    return models


def compute_xgboost_feature_importance(
    pipeline: Pipeline | MultiHeadPipeline,
    X: pd.DataFrame,
    *,
    top_n: int = 20,
    max_rows: int = SHAP_SAMPLE_ROWS,
) -> dict[str, float]:
    xgb_models = _collect_xgboost_models(pipeline)
    if not xgb_models:
        return {}

    sample = X
    if len(X) > max_rows:
        sample = X.sample(n=max_rows, random_state=0)

    aggregated: dict[str, list[float]] = {}
    for _, model in xgb_models:
        contribs, cols = _xgb_contribs(model, sample)
        mean_abs = np.mean(np.abs(contribs[:, :-1]), axis=0)
        for col, value in zip(cols, mean_abs, strict=False):
            aggregated.setdefault(col, []).append(float(value))

    averaged = {col: float(np.mean(vals)) for col, vals in aggregated.items()}
    ranked = sorted(averaged.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return dict(ranked)


def log_xgboost_shap_importance(
    pipeline: Pipeline | MultiHeadPipeline,
    X: pd.DataFrame,
    *,
    top_n: int = 20,
) -> None:
    if not _wandb_active():
        return

    import wandb

    importance = compute_xgboost_feature_importance(pipeline, X, top_n=top_n)
    if not importance:
        return

    table = wandb.Table(columns=["feature", "mean_abs_contribution"])
    for feature, score in importance.items():
        table.add_data(feature, score)
    wandb.log(
        {
            "diagnostics/shap_top_features": table,
            "diagnostics/shap_bar": wandb.plot.bar(
                table,
                "feature",
                "mean_abs_contribution",
                title="Top feature contributions (XGBoost pred_contribs)",
            ),
        }
    )
