from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphapulse.experiments.data import META_MODEL_COLUMN, meta_model_from_benchmarks
from alphapulse.models.base import BaseModel
from alphapulse.pipeline.neutralizer import (
    MetaModelNeutralizer,
    neutralize_against_meta,
)
from alphapulse.pipeline.pipeline import Pipeline
from alphapulse.preprocessors.base import BasePreprocessor


class _IdentityPreprocessor(BasePreprocessor):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BasePreprocessor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class _ConstModel(BaseModel):
    def __init__(self, value: float, name: str = "m") -> None:
        self.value = value
        self.name = name
        self.is_trained = True

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float64)

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> BaseModel:
        return self


def test_neutralize_against_meta_reduces_correlation() -> None:
    rng = np.random.default_rng(0)
    n = 200
    meta = rng.standard_normal(n)
    preds = 0.9 * meta + 0.1 * rng.standard_normal(n)
    eras = pd.Series(np.repeat([f"era_{i}" for i in range(10)], n // 10))
    neutral = neutralize_against_meta(preds, meta, eras=eras, proportion=1.0)
    before = np.corrcoef(preds, meta)[0, 1]
    after = np.corrcoef(neutral, meta)[0, 1]
    assert abs(after) < abs(before)


def test_pipeline_meta_neutralization_in_predict() -> None:
    rng = np.random.default_rng(1)
    n = 80
    X = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    meta = rng.standard_normal(n)
    pipeline = Pipeline(
        preprocessors=[_IdentityPreprocessor()],
        models=[_ConstModel(0.5)],
        feature_columns=list(X.columns),
        meta_neutralize_proportion=0.8,
    )
    preds = pipeline.predict(X, meta_model=meta)
    assert preds.shape == (n,)
    assert np.all(np.isfinite(preds))


def test_meta_neutralizer_optimize_proportion() -> None:
    rng = np.random.default_rng(2)
    n = 120
    eras = pd.Series(np.repeat([f"era_{i}" for i in range(6)], n // 6))
    meta = rng.standard_normal(n)
    y = 0.2 * meta + rng.standard_normal(n) * 0.5
    preds = 0.85 * meta + rng.standard_normal(n) * 0.1
    neutralizer = MetaModelNeutralizer(proportion=0.5)
    optimized = neutralizer.optimize_proportion(
        preds, meta, pd.Series(y), eras, objective="mmc_sharpe"
    )
    assert 0.0 <= optimized <= 1.0


def test_meta_model_from_benchmarks_aligns_to_index() -> None:
    rng = np.random.default_rng(3)
    index = pd.Index(["a", "b", "c"])
    benchmarks = pd.DataFrame(
        {META_MODEL_COLUMN: rng.standard_normal(3)},
        index=index,
    )
    meta = meta_model_from_benchmarks(benchmarks, index)
    assert meta is not None
    assert meta.shape == (3,)


class _FeatureModel(BaseModel):
    def __init__(self) -> None:
        super().__init__("feature")
        self.is_trained = True

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            0.9 * X["f1"].to_numpy(dtype=np.float64)
            + 0.1 * X["f2"].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> BaseModel:
        return self


def test_to_numerai_predict_applies_meta_neutralization() -> None:
    from alphapulse.evaluation.metrics import rank_normalize

    rng = np.random.default_rng(4)
    n = 80
    X = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        }
    )
    meta = rng.standard_normal(n)
    pipeline = Pipeline(
        preprocessors=[_IdentityPreprocessor()],
        models=[_FeatureModel()],
        feature_columns=list(X.columns),
        meta_neutralize_proportion=0.8,
    )
    direct_with = pipeline.predict(X, meta_model=meta)
    direct_without = pipeline.predict(X, meta_model=None)
    assert not np.allclose(direct_with, direct_without)

    predict_fn = pipeline.to_numerai_predict()
    benchmarks = pd.DataFrame({META_MODEL_COLUMN: meta}, index=X.index)
    exported_with = predict_fn(X, benchmarks)["prediction"].to_numpy()
    exported_without = predict_fn(X, pd.DataFrame(index=X.index))[
        "prediction"
    ].to_numpy()
    assert np.allclose(exported_with, rank_normalize(direct_with))
    assert np.allclose(exported_without, rank_normalize(direct_without))
