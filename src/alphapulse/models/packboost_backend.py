from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .packboost_encoding import (
    bin_features_for_packboost,
    default_nfeatsets,
    encode_era_ids,
    q30_predictions_to_float,
    sort_rows_by_era,
)


def packboost_cuda_available() -> bool:
    try:
        require_packboost_cuda(device="cuda")
        return True
    except (ImportError, RuntimeError):
        return False


def require_packboost_cuda(*, device: str = "cuda") -> None:
    if device != "cuda":
        raise ValueError(f"PackboostModel only supports device='cuda', got {device!r}.")
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PackBoost requires PyTorch. Install with: uv sync --extra packboost"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PackBoost CUDA is required but no CUDA device is available."
        )
    try:
        from packboost.core import PackBoost  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PackBoost is not installed. Install with: uv sync --extra packboost"
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            "PackBoost CUDA kernels failed to load. "
            "Ensure ninja and a matching CUDA toolkit are installed."
        ) from exc


def resolve_packboost_device(device: str) -> str:
    require_packboost_cuda(device=device)
    return "cuda"


class PackBoostTrainer:
    def __init__(
        self,
        *,
        device: str = "auto",
        max_depth: int = 7,
        nfolds: int = 8,
        lr: float = 0.07,
        l2: float = 100_000.0,
        nfeatsets: int = 32,
        seed: int = 42,
    ) -> None:
        self.device = resolve_packboost_device(device)
        self.max_depth = int(max_depth)
        self.nfolds = int(nfolds)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.nfeatsets = int(nfeatsets)
        self.seed = int(seed)
        self._model: Any = None

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        era: pd.Series | None = None,
        val_features: pd.DataFrame | None = None,
        val_target: pd.Series | None = None,
        rounds: int,
    ) -> None:
        from packboost.core import PackBoost

        fit_features = features
        fit_target = target
        fit_era = era
        if era is not None:
            fit_features, fit_target, fit_era = sort_rows_by_era(features, target, era)

        x_train = bin_features_for_packboost(fit_features)
        y_train = fit_target.to_numpy(dtype=np.float32)
        n_features = x_train.shape[1]
        nfeatsets = default_nfeatsets(n_features, self.nfeatsets)

        era_ids: np.ndarray | None = None
        if fit_era is not None:
            era_ids = encode_era_ids(fit_era)

        x_val: np.ndarray | None = None
        y_val: np.ndarray | None = None
        if val_features is not None and val_target is not None:
            x_val = bin_features_for_packboost(val_features)
            y_val = val_target.to_numpy(dtype=np.float32)

        model = PackBoost(device=self.device)
        model.fit(
            x_train,
            y_train,
            Xv=x_val,
            Yv=y_val,
            nfolds=self.nfolds,
            rounds=int(rounds),
            max_depth=self.max_depth,
            lr=self.lr,
            L2=self.l2,
            nfeatsets=nfeatsets,
            seed=self.seed,
            era_ids=era_ids,
        )
        self._model = model

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("PackBoostTrainer is not fitted.")
        x_test = bin_features_for_packboost(features)
        pred_q30 = self._model.predict(x_test)
        return q30_predictions_to_float(pred_q30)
