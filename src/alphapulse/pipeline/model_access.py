from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.base import BaseModel
from .multi_target import MultiTargetPipeline
from .multihead import MultiHeadPipeline
from .pipeline import Pipeline
from .row_utils import protected_metadata_frame

PipelineLike = Pipeline | MultiHeadPipeline | MultiTargetPipeline


def iter_trained_models(pipeline: PipelineLike) -> list[BaseModel]:
    if isinstance(pipeline, MultiHeadPipeline):
        return [h.model for h in pipeline.heads]
    if isinstance(pipeline, MultiTargetPipeline):
        return list(pipeline._models.values())
    return list(pipeline.models)


def _preprocess_multitarget(
    pipeline: MultiTargetPipeline, X: pd.DataFrame
) -> pd.DataFrame:
    X_t = X
    for pp in pipeline.preprocessors:
        X_t = pp.transform(X_t)
    return X_t


def model_prediction_map(
    pipeline: PipelineLike,
    X_val: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    if isinstance(pipeline, MultiHeadPipeline):
        X_in = X_val[feature_cols] if feature_cols else X_val
        return {"ensemble": pipeline.predict(X_in)}

    if isinstance(pipeline, MultiTargetPipeline):
        X_in = X_val[feature_cols] if feature_cols else X_val
        X_t = _preprocess_multitarget(pipeline, X_in)
        return {
            target: model.predict(X_t) for target, model in pipeline._models.items()
        }

    X_feat = X_val[feature_cols] if feature_cols else X_val
    era_meta = protected_metadata_frame(X_feat)
    X_t = pipeline._preprocess(X_feat, era_meta)
    X_numeric = X_t.select_dtypes(include=[np.number])
    models = list(pipeline.models)
    if len(models) == 1:
        return {models[0].name: models[0].predict(X_numeric)}
    return {m.name: m.predict(X_numeric) for m in models}


def multitarget_blend_weights(pipeline: MultiTargetPipeline) -> np.ndarray | None:
    if pipeline._weights is None:
        return None
    return np.asarray(pipeline._weights, dtype=np.float64)
