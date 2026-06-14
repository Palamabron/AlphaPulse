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

_ModelList = list[tuple[str, object]]


def _wandb_active() -> bool:
    try:
        import wandb

        return wandb.run is not None
    except ImportError:
        return False


def _collect_by_type(
    pipeline: Pipeline | MultiHeadPipeline,
    model_class: type,
) -> _ModelList:
    results: _ModelList = []
    sources = (
        [h.model for h in pipeline.heads]
        if isinstance(pipeline, MultiHeadPipeline)
        else pipeline.models
    )
    for m in sources:
        if isinstance(m, model_class):
            results.append((m.name, m))
        elif isinstance(m, EraEnsembleModel):
            for sub in m._sub_models:
                if isinstance(sub, model_class):
                    results.append((sub.name, sub))
    return results


def _collect_xgboost_models(pipeline: Pipeline | MultiHeadPipeline) -> _ModelList:
    return _collect_by_type(pipeline, XGBoostModel)


def _collect_lgbm_models(pipeline: Pipeline | MultiHeadPipeline) -> _ModelList:
    from ..models.lightgbm_model import LightGBMModel

    return _collect_by_type(pipeline, LightGBMModel)


def _collect_catboost_models(pipeline: Pipeline | MultiHeadPipeline) -> _ModelList:
    from ..models.catboost_model import CatBoostModel

    return _collect_by_type(pipeline, CatBoostModel)


def _collect_sklearn_tree_models(pipeline: Pipeline | MultiHeadPipeline) -> _ModelList:
    from ..models.sklearn_models import ExtraTreesModel, RandomForestModel

    results: _ModelList = []
    for cls in (RandomForestModel, ExtraTreesModel):
        results.extend(_collect_by_type(pipeline, cls))
    return results


def _xgb_contribs(model: XGBoostModel, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feat = _numeric(X)
    dmat = xgb.DMatrix(feat)
    contribs = model.model.predict(dmat, pred_contribs=True)
    cols = list(feat.columns)
    return np.asarray(contribs, dtype=np.float64), cols


def _lgbm_importance(model: object, feature_cols: list[str]) -> dict[str, float]:
    """Gain-based LightGBM importance, normalized to [0, 1]."""
    booster = getattr(model, "model", None)
    if booster is None:
        return {}
    raw = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    total = float(raw.sum())
    if total <= 0:
        return {}
    result: dict[str, float] = {}
    for name, val in zip(names, raw, strict=False):
        if name in feature_cols:
            result[name] = float(val) / total
    return result


def _catboost_importance(model: object, feature_cols: list[str]) -> dict[str, float]:
    """CatBoost PredictionValuesChange importance, normalized to [0, 1]."""
    cb = getattr(model, "model", None)
    if cb is None:
        return {}
    raw = cb.get_feature_importance()
    names = cb.feature_names_
    total = float(np.sum(raw))
    if total <= 0:
        return {}
    result: dict[str, float] = {}
    for name, val in zip(names, raw, strict=False):
        if name in feature_cols:
            result[name] = float(val) / total
    return result


def _sklearn_tree_importance(
    model: object, feature_cols: list[str]
) -> dict[str, float]:
    """sklearn tree feature_importances_, normalized to [0, 1]."""
    estimator = getattr(model, "model", None)
    if estimator is None:
        return {}
    raw = getattr(estimator, "feature_importances_", None)
    if raw is None:
        return {}
    total = float(np.sum(raw))
    if total <= 0:
        return {}
    return {col: float(v) / total for col, v in zip(feature_cols, raw, strict=False)}


def _aggregate_importance(
    per_model: list[dict[str, float]],
) -> dict[str, float]:
    """Average normalized importance dicts across models."""
    if not per_model:
        return {}
    combined: dict[str, list[float]] = {}
    for imp in per_model:
        for feat, val in imp.items():
            combined.setdefault(feat, []).append(val)
    return {feat: float(np.mean(vals)) for feat, vals in combined.items()}


def compute_universal_feature_importance(
    pipeline: Pipeline | MultiHeadPipeline,
    X: pd.DataFrame,
    *,
    feature_cols: list[str],
    top_n: int = 20,
    max_rows: int = SHAP_SAMPLE_ROWS,
) -> tuple[dict[str, float], str]:
    """Extract feature importance from any supported model type in the pipeline.

    Tries XGBoost (pred_contribs), LightGBM (gain), CatBoost, and sklearn tree
    models. Normalizes each model's scores to [0, 1] and averages across models
    of all types present.

    Args:
        pipeline: Trained pipeline.
        X: Feature DataFrame (pre-preprocessed, numeric).
        feature_cols: Feature column names to report.
        top_n: Maximum features to return.
        max_rows: Row cap for XGBoost pred_contribs (expensive).

    Returns:
        Tuple of (importance_dict sorted descending, model_type_label).
        importance_dict is empty if no supported models are found.
    """
    sample = X if len(X) <= max_rows else X.sample(n=max_rows, random_state=0)

    per_model_imps: list[dict[str, float]] = []
    type_labels: list[str] = []

    xgb_models = _collect_xgboost_models(pipeline)
    for _, m in xgb_models:
        try:
            contribs, cols = _xgb_contribs(m, sample)  # type: ignore[arg-type]
            mean_abs = np.mean(np.abs(contribs[:, :-1]), axis=0)
            total = float(mean_abs.sum())
            if total > 0:
                imp = {
                    c: float(v) / total
                    for c, v in zip(cols, mean_abs, strict=False)
                    if c in feature_cols
                }
                per_model_imps.append(imp)
                type_labels.append("XGBoost")
        except Exception:  # noqa: BLE001, S112
            continue

    lgbm_models = _collect_lgbm_models(pipeline)
    for _, m in lgbm_models:
        imp = _lgbm_importance(m, feature_cols)
        if imp:
            per_model_imps.append(imp)
            type_labels.append("LightGBM")

    cat_models = _collect_catboost_models(pipeline)
    for _, m in cat_models:
        imp = _catboost_importance(m, feature_cols)
        if imp:
            per_model_imps.append(imp)
            type_labels.append("CatBoost")

    sklearn_models = _collect_sklearn_tree_models(pipeline)
    for _, m in sklearn_models:
        imp = _sklearn_tree_importance(m, feature_cols)
        if imp:
            per_model_imps.append(imp)
            type_labels.append("SklearnTree")

    if not per_model_imps:
        return {}, "none"

    averaged = _aggregate_importance(per_model_imps)
    ranked = sorted(averaged.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    label = "+".join(sorted(set(type_labels)))
    return dict(ranked), label


def log_universal_feature_importance(
    pipeline: Pipeline | MultiHeadPipeline,
    X: pd.DataFrame,
    *,
    feature_cols: list[str],
    top_n: int = 20,
) -> dict[str, float]:
    """Log universal feature importance for all supported model types to WandB.

    Args:
        pipeline: Trained pipeline.
        X: Feature DataFrame.
        feature_cols: Feature column names.
        top_n: Number of top features to log.

    Returns:
        The importance dict (empty if WandB is not active or no models found).
    """
    if not _wandb_active():
        return {}

    import wandb

    importance, model_type = compute_universal_feature_importance(
        pipeline, X, feature_cols=feature_cols, top_n=top_n
    )
    if not importance:
        return {}

    table = wandb.Table(columns=["feature", "mean_abs_contribution"])
    for feature, score in importance.items():
        table.add_data(feature, score)
    wandb.log(
        {
            "diagnostics/feature_importance_top": table,
            "diagnostics/feature_importance_bar": wandb.plot.bar(
                table,
                "feature",
                "mean_abs_contribution",
                title=f"Top feature importance ({model_type})",
            ),
        }
    )
    if wandb.run is not None:
        wandb.run.summary["diagnostics/feature_importance_model_type"] = model_type
    return importance
