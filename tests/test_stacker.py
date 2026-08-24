from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphapulse.models.base import BaseModel
from alphapulse.models.factory import ModelFactory
from alphapulse.pipeline.stacker import Stacker
from alphapulse.validation.purged_cv import PurgedEraCV


class _RecordingModel(BaseModel):
    validation_lengths: list[int | None] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        self.validation_lengths.append(None if X_val is None else len(X_val))
        self.is_trained = True
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            X.select_dtypes(include=[np.number]).iloc[:, 0].to_numpy(),
            dtype=np.float64,
        )

    def save(self, path: Path) -> None:
        return None

    def load(self, path: Path) -> BaseModel:
        return self


def test_collect_oof_does_not_use_test_fold_for_early_stopping(
    monkeypatch: Any,
) -> None:
    rng = np.random.default_rng(0)
    eras = pd.Series(np.repeat([f"era_{i:03d}" for i in range(20)], 4))
    X = pd.DataFrame(rng.standard_normal((len(eras), 3)), columns=list("abc"))
    y = pd.Series(rng.standard_normal(len(eras)))
    cv = PurgedEraCV(n_splits=2, n_purge=1, n_embargo=0, min_train_eras=5)
    _RecordingModel.validation_lengths = []
    monkeypatch.setattr(
        ModelFactory,
        "suggest_fixed",
        lambda self, model_type: _RecordingModel("recording"),
    )

    Stacker(X=X, y=y, eras=eras, cv=cv).collect_oof([{"model_type": "LightGBM"}])

    assert _RecordingModel.validation_lengths
    assert all(length is None for length in _RecordingModel.validation_lengths)


def test_model_factory_accepts_legacy_lowercase_model_names() -> None:
    model = ModelFactory().suggest_fixed("lightgbm", {"n_subs": 2})
    assert model.name.startswith("EraEnsemble")
