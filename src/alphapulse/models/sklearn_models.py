from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd

from .base import BaseModel


def _numeric(X: pd.DataFrame) -> pd.DataFrame:
    return X.select_dtypes(include=[np.number])


def _save_sklearn(model: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)


def _load_sklearn(path: Path) -> Any:
    with open(path, "rb") as f:
        return cloudpickle.load(f)


class RandomForestModel(BaseModel):
    """sklearn RandomForestRegressor.

    Bagged fully-grown trees give predictions that are decorrelated from
    gradient boosting models, making this a high-diversity ensemble member.
    min_samples_leaf prevents per-stock overfitting in era-structured data.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        name: str | None = "RandomForest",
    ) -> None:
        super().__init__(name)
        self.params: dict[str, Any] = params or {
            "n_estimators": 300,
            "min_samples_leaf": 200,
            "max_features": 0.3,
            "n_jobs": -1,
            "random_state": 42,
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        from sklearn.ensemble import RandomForestRegressor

        feat = _numeric(X_train)
        if feat.shape[1] == 0:
            raise ValueError("RandomForestModel: no numeric feature columns found.")
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(feat, y_train)
        self.is_trained = True
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained!")
        return np.asarray(self.model.predict(_numeric(X)), dtype=np.float64)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        _save_sklearn(self.model, path)

    def load(self, path: Path) -> "RandomForestModel":
        self.model = _load_sklearn(path)
        self.is_trained = True
        return self


class ExtraTreesModel(BaseModel):
    """sklearn ExtraTreesRegressor.

    Splits are chosen at random (not optimally) which produces trees that are
    even more decorrelated than RandomForest — maximising diversity in an
    ensemble at the cost of slightly higher bias per tree.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        name: str | None = "ExtraTrees",
    ) -> None:
        super().__init__(name)
        self.params: dict[str, Any] = params or {
            "n_estimators": 300,
            "min_samples_leaf": 200,
            "max_features": 0.3,
            "n_jobs": -1,
            "random_state": 42,
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        from sklearn.ensemble import ExtraTreesRegressor

        feat = _numeric(X_train)
        if feat.shape[1] == 0:
            raise ValueError("ExtraTreesModel: no numeric feature columns found.")
        self.model = ExtraTreesRegressor(**self.params)
        self.model.fit(feat, y_train)
        self.is_trained = True
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained!")
        return np.asarray(self.model.predict(_numeric(X)), dtype=np.float64)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        _save_sklearn(self.model, path)

    def load(self, path: Path) -> "ExtraTreesModel":
        self.model = _load_sklearn(path)
        self.is_trained = True
        return self


class RidgeModel(BaseModel):
    """sklearn Ridge (L2-regularised linear regression).

    Captures linear combinations of features that gradient-boosted trees
    can miss.  Pairs well with a StandardScaler or RobustScaler preprocessor.
    High alpha (default 100) is appropriate for Numerai's ~1 200-feature space.
    """

    def __init__(
        self,
        alpha: float = 100.0,
        name: str | None = "Ridge",
    ) -> None:
        super().__init__(name)
        self.alpha = alpha

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        from sklearn.linear_model import Ridge

        feat = _numeric(X_train)
        if feat.shape[1] == 0:
            raise ValueError("RidgeModel: no numeric feature columns found.")
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(feat, y_train)
        self.is_trained = True
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained!")
        return np.asarray(self.model.predict(_numeric(X)), dtype=np.float64)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        _save_sklearn(self.model, path)

    def load(self, path: Path) -> "RidgeModel":
        self.model = _load_sklearn(path)
        self.is_trained = True
        return self
