from typing import Any, Self

import numpy as np
import pandas as pd

from .base import BasePreprocessor

DEFAULT_LATENT_DIM = 64
DEFAULT_HIDDEN_DIM = 256
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_SEED = 42
_STD_EPS = 1e-8


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "AutoencoderPreprocessor requires torch. "
            "Install with: pip install 'alphapulse[deep]'"
        ) from exc
    return torch


class AutoencoderPreprocessor(BasePreprocessor):
    _medians: pd.Series
    _mean: np.ndarray
    _std: np.ndarray
    _encoder: Any
    _device: Any

    def __init__(
        self,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        device: str | None = None,
        seed: int = DEFAULT_SEED,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if epochs < 1:
            raise ValueError(f"epochs must be positive, got {epochs}")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.seed = seed
        self._numeric_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        torch = _require_torch()
        self._numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not self._numeric_cols:
            raise ValueError("AutoencoderPreprocessor: no numeric columns found.")
        if self.latent_dim >= len(self._numeric_cols):
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be smaller than the number "
                f"of numeric columns ({len(self._numeric_cols)})."
            )

        values = X[self._numeric_cols].astype(np.float64)
        self._medians = values.median()
        arr = values.fillna(self._medians).to_numpy(dtype=np.float32)
        self._mean = arr.mean(axis=0)
        self._std = arr.std(axis=0) + _STD_EPS
        arr = (arr - self._mean) / self._std

        torch.manual_seed(self.seed)
        self._device = self._resolve_device(torch)
        encoder, model = self._build_model(torch, arr.shape[1])
        self._train_model(torch, model, arr)
        encoder.eval()
        self._encoder = encoder
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted!")
        torch = _require_torch()
        values = X[self._numeric_cols].astype(np.float64).fillna(self._medians)
        arr = (values.to_numpy(dtype=np.float32) - self._mean) / self._std
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(arr), self.batch_size):
                batch = torch.from_numpy(arr[start : start + self.batch_size])
                outputs.append(self._encoder(batch.to(self._device)).cpu().numpy())
        out = (
            np.concatenate(outputs)
            if outputs
            else np.empty((0, self.latent_dim), dtype=np.float32)
        )
        cols = [f"ae_{i}" for i in range(self.latent_dim)]
        return pd.DataFrame(out.astype(np.float64), columns=cols, index=X.index)

    def _resolve_device(self, torch: Any) -> Any:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, torch: Any, n_features: int) -> tuple[Any, Any]:
        encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.latent_dim),
        )
        decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, n_features),
        )
        model = torch.nn.Sequential(encoder, decoder).to(self._device)
        return encoder, model

    def _train_model(self, torch: Any, model: Any, arr: np.ndarray) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()
        data = torch.from_numpy(arr)
        rng = np.random.default_rng(self.seed)
        model.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(arr))
            for start in range(0, len(arr), self.batch_size):
                batch = data[order[start : start + self.batch_size]].to(self._device)
                optimizer.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optimizer.step()
