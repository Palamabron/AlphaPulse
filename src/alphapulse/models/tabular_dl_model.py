from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .base import BaseModel

_PYTORCH_TABULAR_AVAILABLE = False
try:
    import pytorch_tabular  # noqa: F401

    _PYTORCH_TABULAR_AVAILABLE = True
except ImportError:
    pass


def _check_pytorch_tabular() -> None:
    if not _PYTORCH_TABULAR_AVAILABLE:
        raise ImportError(
            "pytorch_tabular is required for TabularDLModel. "
            "Install with: pip install 'alphapulse[deep]'"
        )


class TabularDLModel(BaseModel):
    ARCHITECTURES = ("ft_transformer", "mlp")

    def __init__(
        self,
        architecture: Literal["ft_transformer", "mlp"] = "ft_transformer",
        dl_params: dict[str, Any] | None = None,
        trainer_params: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        _check_pytorch_tabular()
        super().__init__(name=name or f"DL_{architecture}")
        if architecture not in self.ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Choose from {self.ARCHITECTURES}"
            )
        self.architecture = architecture
        self.dl_params = dl_params or {}
        self.trainer_params = trainer_params or {}
        self._target_col = "__target__"
        self._feature_cols: list[str] = []

    def _build_model_config(self) -> Any:
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
        from pytorch_tabular.models import (
            CategoryEmbeddingModelConfig,
            FTTransformerConfig,
        )

        epochs = self.trainer_params.get("epochs", 50)
        batch_size = self.trainer_params.get("batch_size", 1024)
        lr = self.dl_params.get("learning_rate", 1e-3)

        trainer_config = TrainerConfig(
            max_epochs=epochs,
            batch_size=batch_size,
            accelerator="auto",
            early_stopping="valid_loss",
            early_stopping_patience=self.trainer_params.get("patience", 5),
            checkpoints="valid_loss",
            checkpoints_path="__pt_checkpoints__",
            progress_bar="none",
            load_best=True,
            trainer_kwargs={"enable_model_summary": False},
        )
        data_config = DataConfig(
            target=[self._target_col],
            continuous_cols=self._feature_cols,
            categorical_cols=[],
        )
        optimizer_config = OptimizerConfig(
            optimizer="AdamW",
            optimizer_params={"weight_decay": self.dl_params.get("weight_decay", 1e-4)},
            lr_scheduler="ReduceLROnPlateau",
            lr_scheduler_params={"patience": 3, "factor": 0.5},
        )

        if self.architecture == "ft_transformer":
            model_config = FTTransformerConfig(
                task="regression",
                input_embed_dim=self.dl_params.get("embed_dim", 32),
                num_heads=self.dl_params.get("num_heads", 4),
                num_attn_blocks=self.dl_params.get("num_attn_blocks", 3),
                attn_dropout=self.dl_params.get("dropout", 0.1),
                ff_dropout=self.dl_params.get("dropout", 0.1),
                learning_rate=lr,
            )
        else:
            model_config = CategoryEmbeddingModelConfig(
                task="regression",
                layers=self.dl_params.get("layers", "128-64-32"),
                dropout=self.dl_params.get("dropout", 0.1),
                activation="ReLU",
                learning_rate=lr,
            )

        return data_config, trainer_config, optimizer_config, model_config

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        _check_pytorch_tabular()
        from pytorch_tabular import TabularModel

        self._feature_cols = list(X_train.columns)

        train_df = X_train.copy()
        train_df[self._target_col] = y_train.values

        val_df = None
        if X_val is not None and y_val is not None:
            val_df = X_val.copy()
            val_df[self._target_col] = y_val.values

        data_config, trainer_config, optimizer_config, model_config = (
            self._build_model_config()
        )

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=False,
        )
        tabular_model.fit(train=train_df, validation=val_df)

        self.model = tabular_model
        self.is_trained = True

        metrics: dict[str, float] = {}
        if val_df is not None:
            result = tabular_model.evaluate(val_df, verbose=False)
            if isinstance(result, list) and len(result) > 0:
                result_dict = result[0] if isinstance(result[0], dict) else {}
            elif isinstance(result, dict):
                result_dict = result
            else:
                result_dict = {}
            for k, v in result_dict.items():
                metrics[f"eval_{k}"] = float(v)

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained!")
        df = X[self._feature_cols].copy()
        df[self._target_col] = 0.0
        pred_df = self.model.predict(df)
        col = pred_df.columns[0]
        return np.asarray(pred_df[col].values, dtype=np.float64)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: Path) -> "TabularDLModel":
        _check_pytorch_tabular()
        from pytorch_tabular import TabularModel

        self.model = TabularModel.load_model(str(path))
        self.is_trained = True
        return self
