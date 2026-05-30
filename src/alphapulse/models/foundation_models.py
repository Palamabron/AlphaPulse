from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseModel, _numeric
from .sklearn_models import _load_sklearn, _save_sklearn


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
            raise ValueError(f"{self.name}: no numeric feature columns found.")
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
        self._require_trained()
        return np.asarray(self.model.predict(_numeric(X)), dtype=np.float64)

    def save(self, path: Path) -> None:
        self._require_trained()
        _save_sklearn(self.model, path)

    def load(self, path: Path) -> "TabPFNModel":
        self.model = _load_sklearn(path)
        self.is_trained = True
        return self


class TabPFN3Model(BaseModel):
    """TabPFN v3 regression via in-context learning (local OSS).

    Requires: pip install 'alphapulse[foundation]'

    Constraints:
        - Up to ~1M training samples (TabPFN v3).
        - GPU recommended for large datasets.
    """

    def __init__(
        self,
        model_path: str = "auto",
        n_estimators: int = 8,
        device: str | None = None,
        ignore_pretraining_limits: bool = False,
        random_state: int = 42,
        name: str | None = "TabPFN3",
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state

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
            raise ValueError(f"{self.name}: no numeric feature columns found.")
        init_kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_estimators": self.n_estimators,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "random_state": self.random_state,
        }
        if self.device is not None:
            init_kwargs["device"] = self.device
        self.model = TabPFNRegressor(**init_kwargs)
        self.model.fit(feat, y_train)
        self.is_trained = True
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return np.asarray(self.model.predict(_numeric(X)), dtype=np.float64)

    def save(self, path: Path) -> None:
        self._require_trained()
        _save_sklearn(self.model, path)

    def load(self, path: Path) -> "TabPFN3Model":
        self.model = _load_sklearn(path)
        self.is_trained = True
        return self


class TabPFN3ReasoningModel(BaseModel):
    """TabPFN v3 regression via Prior Labs API with reasoning mode.

    Requires: pip install 'alphapulse[foundation-api]' and TABPFN_API_KEY.

    Note: fits are API-metered and slower than local TabPFN3Model.
    """

    def __init__(
        self,
        thinking_mode: bool = True,
        thinking_effort: str = "medium",
        thinking_timeout_s: int = 300,
        thinking_metric: str = "rmse",
        name: str | None = "TabPFN3Reasoning",
    ) -> None:
        super().__init__(name)
        self.thinking_mode = thinking_mode
        self.thinking_effort = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.thinking_metric = thinking_metric

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        import os

        try:
            from tabpfn_client import TabPFNRegressor
        except ImportError as exc:
            raise ImportError(
                "TabPFN3ReasoningModel requires tabpfn-client. "
                "Install with: pip install 'alphapulse[foundation-api]'"
            ) from exc

        if not os.environ.get("TABPFN_API_KEY"):
            raise ValueError(
                "TabPFN3ReasoningModel requires TABPFN_API_KEY environment variable."
            )

        feat = _numeric(X_train)
        if feat.shape[1] == 0:
            raise ValueError(f"{self.name}: no numeric feature columns found.")
        init_kwargs: dict[str, Any] = {
            "thinking_mode": self.thinking_mode,
            "thinking_effort": self.thinking_effort,
            "thinking_timeout_s": self.thinking_timeout_s,
            "thinking_metric": self.thinking_metric,
        }
        self.model = TabPFNRegressor(**init_kwargs)
        self.model.fit(feat, y_train)
        self.is_trained = True
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        return np.asarray(self.model.predict(_numeric(X)), dtype=np.float64)

    def save(self, path: Path) -> None:
        self._require_trained()
        _save_sklearn(self.model, path)

    def load(self, path: Path) -> "TabPFN3ReasoningModel":
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
            raise ValueError(f"{self.name}: no numeric feature columns found.")
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
        self._require_trained()
        return np.asarray(self.model.predict(_numeric(X)), dtype=np.float64)

    def save(self, path: Path) -> None:
        self._require_trained()
        _save_sklearn(self.model, path)

    def load(self, path: Path) -> "TabICLModel":
        self.model = _load_sklearn(path)
        self.is_trained = True
        return self
