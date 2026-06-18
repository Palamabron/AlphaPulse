from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..preprocessors.autoencoder import AutoencoderPreprocessor
from ..preprocessors.base import BasePreprocessor
from ..preprocessors.compression import PCAPreprocessor, TruncatedSVDPreprocessor
from .base import BaseModel, _numeric
from .sklearn_models import _load_sklearn, _save_sklearn

COMPRESSION_METHODS = ("pca", "svd", "autoencoder")
DEFAULT_COMPRESSION = "pca"
DEFAULT_SEED = 42
DEFAULT_PREDICT_CHUNK_ROWS = 256
PREDICT_CHUNK_ROWS = DEFAULT_PREDICT_CHUNK_ROWS

TABPFN_MAX_TRAIN_ROWS = 10_000
TABPFN_MAX_FEATURES = 500
TABPFN3_MAX_TRAIN_ROWS = 100_000
TABPFN3_MAX_FEATURES = 2_000
TABICL_MAX_TRAIN_ROWS = 60_000
TABICL_MAX_FEATURES = 500
TABPFN_PREDICT_CHUNK_ROWS = 256
TABPFN3_PREDICT_CHUNK_ROWS = 128
TABICL_PREDICT_CHUNK_ROWS = 2_048


def _build_compressor(
    method: str,
    n_components: int,
    seed: int,
    *,
    epochs: int = 20,
    device: str | None = None,
) -> BasePreprocessor:
    if method == "pca":
        return PCAPreprocessor(n_components=n_components, random_state=seed)
    if method == "svd":
        return TruncatedSVDPreprocessor(n_components=n_components, random_state=seed)
    if method == "autoencoder":
        return AutoencoderPreprocessor(
            latent_dim=n_components, seed=seed, epochs=epochs, device=device
        )
    raise ValueError(
        f"Unknown compression method: {method!r}. "
        f"Expected one of {COMPRESSION_METHODS}."
    )


class FoundationModel(BaseModel):
    """Base for in-context tabular foundation models (TabPFN/TabICL).

    These models have hard limits on context size, so the wrapper makes them
    work on Numerai-scale data by (1) randomly subsampling training rows to
    ``max_train_rows``, (2) compressing features to at most ``max_features``
    columns via PCA/SVD/autoencoder when the input is wider, and
    (3) predicting in chunks to bound memory.
    """

    def __init__(
        self,
        *,
        max_train_rows: int,
        max_features: int,
        compression: str | None = DEFAULT_COMPRESSION,
        compression_components: int | None = None,
        compression_epochs: int = 20,
        compression_device: str | None = None,
        predict_chunk_rows: int = DEFAULT_PREDICT_CHUNK_ROWS,
        seed: int = DEFAULT_SEED,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        if max_train_rows < 1:
            raise ValueError(f"max_train_rows must be positive, got {max_train_rows}")
        if max_features < 1:
            raise ValueError(f"max_features must be positive, got {max_features}")
        if predict_chunk_rows < 1:
            raise ValueError(
                f"predict_chunk_rows must be positive, got {predict_chunk_rows}"
            )
        if compression is not None and compression not in COMPRESSION_METHODS:
            raise ValueError(
                f"Unknown compression method: {compression!r}. "
                f"Expected one of {COMPRESSION_METHODS} or None."
            )
        self.max_train_rows = max_train_rows
        self.max_features = max_features
        self.compression = compression
        self.compression_components = compression_components
        self.compression_epochs = compression_epochs
        self.compression_device = compression_device
        self.predict_chunk_rows = predict_chunk_rows
        self.seed = seed
        self._compressor: BasePreprocessor | None = None
        self._medians: pd.Series | None = None

    @abstractmethod
    def _make_regressor(self) -> Any: ...

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        regressor = self._make_regressor()
        eras = X_train["era"] if "era" in X_train.columns else None
        feat, y = self._prepare_train(X_train, y_train, eras=eras)
        regressor.fit(feat, y)
        self.model = regressor
        self.is_trained = True
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        feat = self._prepare_predict(X)
        chunk_rows = self.predict_chunk_rows
        chunks = [
            np.asarray(
                self.model.predict(feat.iloc[start : start + chunk_rows]),
                dtype=np.float64,
            )
            for start in range(0, len(feat), chunk_rows)
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)

    def save(self, path: Path) -> None:
        self._require_trained()
        payload = {
            "model": self.model,
            "compressor": self._compressor,
            "medians": self._medians,
        }
        _save_sklearn(payload, path)

    def load(self, path: Path) -> "FoundationModel":
        payload = _load_sklearn(path)
        self.model = payload["model"]
        self._compressor = payload["compressor"]
        self._medians = payload["medians"]
        self.is_trained = True
        return self

    def _prepare_train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eras: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        feat = _numeric(X_train)
        if feat.shape[1] == 0:
            raise ValueError(f"{self.name}: no numeric feature columns found.")
        feat, y = self._subsample(feat, y_train, eras=eras)
        self._medians = feat.median()
        feat = feat.fillna(self._medians)
        feat = self._fit_compressor(feat, y)
        return feat, y

    def _subsample_random(
        self, feat: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(feat), size=self.max_train_rows, replace=False)
        idx.sort()
        return feat.iloc[idx], y.iloc[idx]

    def _subsample_era_stratified(
        self, feat: pd.DataFrame, y: pd.Series, eras: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        era_vals = np.asarray(eras)
        unique_eras = np.unique(era_vals)
        n_eras = len(unique_eras)
        if n_eras == 0:
            return self._subsample_random(feat, y)
        per_era = max(1, self.max_train_rows // n_eras)
        rng = np.random.default_rng(self.seed)
        selected_idx: list[int] = []
        for era in unique_eras:
            era_idx = np.flatnonzero(era_vals == era)
            if len(era_idx) <= per_era:
                selected_idx.extend(era_idx.tolist())
            else:
                chosen = rng.choice(era_idx, size=per_era, replace=False)
                selected_idx.extend(sorted(chosen.tolist()))
        if len(selected_idx) > self.max_train_rows:
            selected_idx = sorted(
                rng.choice(
                    selected_idx, size=self.max_train_rows, replace=False
                ).tolist()
            )
        return feat.iloc[selected_idx], y.iloc[selected_idx]

    def _subsample(
        self,
        feat: pd.DataFrame,
        y: pd.Series,
        eras: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if len(feat) <= self.max_train_rows:
            return feat, y
        if eras is not None and len(eras) == len(feat):
            return self._subsample_era_stratified(feat, y, eras)
        return self._subsample_random(feat, y)

    def _fit_compressor(self, feat: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if self.compression is None or feat.shape[1] <= self.max_features:
            self._compressor = None
            return feat
        n_components = self.compression_components or self.max_features
        n_components = min(n_components, feat.shape[1] - 1, len(feat))
        self._compressor = _build_compressor(
            self.compression,
            n_components,
            self.seed,
            epochs=self.compression_epochs,
            device=self.compression_device,
        )
        return self._compressor.fit_transform(feat, y)

    def _prepare_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        feat = _numeric(X)
        if self._medians is not None:
            feat = feat.reindex(columns=self._medians.index).fillna(self._medians)
        if self._compressor is not None:
            feat = self._compressor.transform(feat)
        return feat


class TabPFNModel(FoundationModel):
    """TabPFN v2 regression via in-context learning.

    Requires: pip install 'alphapulse[foundation]'
    """

    def __init__(
        self,
        n_estimators: int = 8,
        device: str | None = None,
        ignore_pretraining_limits: bool = False,
        max_train_rows: int = TABPFN_MAX_TRAIN_ROWS,
        max_features: int = TABPFN_MAX_FEATURES,
        compression: str | None = DEFAULT_COMPRESSION,
        compression_components: int | None = None,
        compression_epochs: int = 20,
        compression_device: str | None = None,
        predict_chunk_rows: int = TABPFN_PREDICT_CHUNK_ROWS,
        seed: int = DEFAULT_SEED,
        name: str | None = "TabPFN",
    ) -> None:
        super().__init__(
            max_train_rows=max_train_rows,
            max_features=max_features,
            compression=compression,
            compression_components=compression_components,
            compression_epochs=compression_epochs,
            compression_device=compression_device,
            predict_chunk_rows=predict_chunk_rows,
            seed=seed,
            name=name,
        )
        self.n_estimators = n_estimators
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits

    def _make_regressor(self) -> Any:
        from tabpfn import TabPFNRegressor

        init_kwargs: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
        }
        if self.device is not None:
            init_kwargs["device"] = self.device
        return TabPFNRegressor(**init_kwargs)


class TabPFN3Model(FoundationModel):
    """TabPFN v3 regression via in-context learning (local OSS).

    Requires: pip install 'alphapulse[foundation]'
    """

    def __init__(
        self,
        model_path: str = "auto",
        n_estimators: int = 8,
        device: str | None = None,
        ignore_pretraining_limits: bool = False,
        random_state: int = DEFAULT_SEED,
        max_train_rows: int = TABPFN3_MAX_TRAIN_ROWS,
        max_features: int = TABPFN3_MAX_FEATURES,
        compression: str | None = DEFAULT_COMPRESSION,
        compression_components: int | None = None,
        compression_epochs: int = 20,
        compression_device: str | None = None,
        predict_chunk_rows: int = TABPFN3_PREDICT_CHUNK_ROWS,
        seed: int = DEFAULT_SEED,
        name: str | None = "TabPFN3",
    ) -> None:
        super().__init__(
            max_train_rows=max_train_rows,
            max_features=max_features,
            compression=compression,
            compression_components=compression_components,
            compression_epochs=compression_epochs,
            compression_device=compression_device,
            predict_chunk_rows=predict_chunk_rows,
            seed=seed,
            name=name,
        )
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state

    def _make_regressor(self) -> Any:
        from tabpfn import TabPFNRegressor

        init_kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_estimators": self.n_estimators,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "random_state": self.random_state,
        }
        if self.device is not None:
            init_kwargs["device"] = self.device
        return TabPFNRegressor(**init_kwargs)


class TabICLModel(FoundationModel):
    """TabICL v2 regression via in-context learning.

    Requires: pip install 'alphapulse[foundation]'
    """

    def __init__(
        self,
        n_estimators: int = 8,
        device: str | None = None,
        kv_cache: bool = False,
        batch_size: int = 8,
        random_state: int = DEFAULT_SEED,
        max_train_rows: int = TABICL_MAX_TRAIN_ROWS,
        max_features: int = TABICL_MAX_FEATURES,
        compression: str | None = DEFAULT_COMPRESSION,
        compression_components: int | None = None,
        compression_epochs: int = 20,
        compression_device: str | None = None,
        predict_chunk_rows: int = TABICL_PREDICT_CHUNK_ROWS,
        seed: int = DEFAULT_SEED,
        name: str | None = "TabICL",
    ) -> None:
        super().__init__(
            max_train_rows=max_train_rows,
            max_features=max_features,
            compression=compression,
            compression_components=compression_components,
            compression_epochs=compression_epochs,
            compression_device=compression_device,
            predict_chunk_rows=predict_chunk_rows,
            seed=seed,
            name=name,
        )
        self.n_estimators = n_estimators
        self.device = device
        self.kv_cache = kv_cache
        self.batch_size = batch_size
        self.random_state = random_state

    def _make_regressor(self) -> Any:
        from tabicl import TabICLRegressor

        init_kwargs: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "kv_cache": self.kv_cache,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }
        if self.device is not None:
            init_kwargs["device"] = self.device
        return TabICLRegressor(**init_kwargs)
