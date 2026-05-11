from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseModel
from .sklearn_models import _load_sklearn, _numeric, _save_sklearn


class TabPFNModel(BaseModel):
    """TabPFN v2 regression via in-context learning.

    A pre-trained foundation model that learns from context at inference time.
    No gradient-based training occurs — fit() stores the data and predict()
    performs in-context learning over it.

    Requires: pip install 'alphapulse[foundation]'

    Constraints:
        - Up to 50 000 training samples and 2 000 features (TabPFN v2).
        - GPU recommended; CPU feasible only for small datasets (≲1 000 rows).
    """

    def __init__(
        self,
        n_estimators: int = 8,
        device: str | None = None,
        ignore_pretraining_limits: bool = False,
        name: str | None = "TabPFN",
    ) -> None:
        super().__init__(name)
        self.n_estimators = n_estimators
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        from tabpfn import TabPFNRegressor

        feat = _numeric(X_train)
        if feat.shape[1] == 0:
            raise ValueError("TabPFNModel: no numeric feature columns found.")
        init_kwargs: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
        }
        if self.device is not None:
            init_kwargs["device"] = self.device
        self.model = TabPFNRegressor(**init_kwargs)
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

    def load(self, path: Path) -> "TabPFNModel":
        self.model = _load_sklearn(path)
        self.is_trained = True
        return self


class TabICLModel(BaseModel):
    """TabICL v2 regression via in-context learning.

    A pre-trained tabular foundation model that scales to 600 K+ rows via
    CPU/disk offloading. Like TabPFN, fit() stores the context and learning
    occurs during predict().

    Requires: pip install 'alphapulse[foundation]'
    """

    def __init__(
        self,
        n_estimators: int = 8,
        device: str | None = None,
        kv_cache: bool = False,
        batch_size: int = 8,
        random_state: int = 42,
        name: str | None = "TabICL",
    ) -> None:
        super().__init__(name)
        self.n_estimators = n_estimators
        self.device = device
        self.kv_cache = kv_cache
        self.batch_size = batch_size
        self.random_state = random_state

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        from tabicl import TabICLRegressor

        feat = _numeric(X_train)
        if feat.shape[1] == 0:
            raise ValueError("TabICLModel: no numeric feature columns found.")
        init_kwargs: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "kv_cache": self.kv_cache,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }
        if self.device is not None:
            init_kwargs["device"] = self.device
        self.model = TabICLRegressor(**init_kwargs)
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

    def load(self, path: Path) -> "TabICLModel":
        self.model = _load_sklearn(path)
        self.is_trained = True
        return self
