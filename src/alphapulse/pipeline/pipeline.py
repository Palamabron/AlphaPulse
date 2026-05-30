from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd

from ..evaluation.metrics import rank_normalize
from ..models.base import BaseModel
from ..preprocessors.base import BasePreprocessor, TrainEvalPreprocessor
from .ensemble import EnsembleStrategy
from .neutralizer import FeatureNeutralizer
from .row_utils import (
    filter_invalid_rows,
    filter_nan_rows,
    invalid_row_mask,
    protected_metadata_frame,
    reattach_protected_columns,
    safe_median,
)


class Pipeline:
    """End-to-end ML pipeline: preprocessing, model training, and ensembling.

    Args:
        preprocessors: Ordered list of preprocessing steps applied before
            model training and prediction.
        model: A single model instance (mutually exclusive with *models*).
        models: Multiple model instances for ensemble pipelines.
        feature_columns: Explicit feature column names. Inferred from the
            training DataFrame if *None*.
        ensemble_method: How to combine multi-model predictions.
        ensemble_params: Parameters forwarded to ``EnsembleStrategy``
            (e.g. ``{"weights": [0.6, 0.4]}`` for weighted ensembling).
        benchmark_blend_weight: Fraction to blend Numerai benchmark
            predictions into the final output (0.0 = no blending).
        neutralize_proportion: Proportion of feature exposure to remove from
            predictions (0.0 = disabled). Applied per era when eras are
            provided to ``predict()``. Rank-normalizes output after
            neutralization.
        neutralize_features: Feature columns used as neutralization basis.
            Defaults to all feature columns when *None*.
    """

    def __init__(
        self,
        preprocessors: list[BasePreprocessor],
        model: BaseModel | None = None,
        models: list[BaseModel] | None = None,
        feature_columns: list[str] | None = None,
        ensemble_method: Literal["single", "weighted", "stacking"] = "single",
        ensemble_params: dict[str, Any] | None = None,
        benchmark_blend_weight: float = 0.0,
        neutralize_proportion: float = 0.0,
        neutralize_features: list[str] | None = None,
    ) -> None:
        if model is not None and models is not None:
            raise ValueError("Provide either model or models, not both.")
        self.preprocessors = preprocessors
        self.model: BaseModel | None
        if model is not None:
            self.model = model
            self.models = [model]
        elif models is not None:
            self.model = None
            self.models = models
        else:
            raise ValueError("Either model or models must be provided.")
        if not self.models:
            raise ValueError("Either model or models must be provided.")
        self.feature_columns = feature_columns
        self.benchmark_blend_weight = benchmark_blend_weight
        self._ensemble = EnsembleStrategy(
            method=ensemble_method, params=ensemble_params or {}
        )
        self.neutralize_proportion = float(neutralize_proportion)
        self.neutralize_features = neutralize_features
        self._neutralizer: FeatureNeutralizer | None = (
            FeatureNeutralizer(proportion=self.neutralize_proportion)
            if self.neutralize_proportion > 0.0
            else None
        )

    @property
    def ensemble_method(self) -> str:
        return self._ensemble.method

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **model_train_kwargs: Any,
    ) -> dict[str, float]:
        """Fit preprocessors and train all models.

        Args:
            X: Training features.
            y: Training targets.
            X_val: Optional validation features for early stopping / stacking.
            y_val: Optional validation targets.
            **model_train_kwargs: Forwarded to each model's ``train()``
                (e.g. ``n_rounds``, ``early_stopping_rounds``).

        Returns:
            Dictionary of training metrics reported by the model(s).
        """
        if self.feature_columns is None:
            self.feature_columns = list(X.columns)

        X, y = filter_invalid_rows(X, y)
        if len(X) == 0:
            raise ValueError("No valid training rows after filtering invalid values")

        era_meta = protected_metadata_frame(X)
        X_fit = X
        for pp in self.preprocessors:
            if isinstance(pp, TrainEvalPreprocessor):
                pp.train()
            pp.fit(X_fit, y)
            X_fit = pp.transform(X_fit)
            X_fit = reattach_protected_columns(X_fit, era_meta)

        X_fit, y = filter_nan_rows(X_fit, y)
        if len(X_fit) == 0:
            raise ValueError("No valid training rows after preprocessing")

        X_val_t: pd.DataFrame | None = None
        if X_val is not None:
            X_val, y_val = filter_invalid_rows(X_val, y_val)
            if len(X_val) == 0:
                X_val_t = None
                y_val = None
            else:
                val_era_meta = protected_metadata_frame(X_val)
                X_val_t = X_val
                for pp in self.preprocessors:
                    if isinstance(pp, TrainEvalPreprocessor):
                        pp.eval()
                    X_val_t = pp.transform(X_val_t)
                    X_val_t = reattach_protected_columns(X_val_t, val_era_meta)

                X_val_t, y_val = filter_nan_rows(X_val_t, y_val)
                if len(X_val_t) == 0:
                    X_val_t = None
                    y_val = None

        if len(self.models) == 1:
            metrics = self.models[0].train(
                X_fit, y, X_val=X_val_t, y_val=y_val, **model_train_kwargs
            )
        else:
            metrics = {}
            for m in self.models:
                m_metrics = m.train(
                    X_fit, y, X_val=X_val_t, y_val=y_val, **model_train_kwargs
                )
                for k, v in m_metrics.items():
                    metrics[f"{m.name}_{k}"] = v

            def get_val_preds() -> np.ndarray:
                return np.column_stack([m.predict(X_val_t) for m in self.models])

            self._ensemble.fit(
                n_models=len(self.models),
                get_val_predictions=get_val_preds if X_val_t is not None else None,
                y_val=y_val,
            )

        return metrics

    def _preprocess(
        self, X: pd.DataFrame, era_meta: pd.DataFrame | None
    ) -> pd.DataFrame:
        X_t = X
        for pp in self.preprocessors:
            X_t = pp.transform(X_t)
            X_t = reattach_protected_columns(X_t, era_meta)
        return X_t

    def _infer(self, X_t: pd.DataFrame) -> np.ndarray:
        if len(self.models) == 1:
            return self.models[0].predict(X_t)
        preds = np.column_stack([m.predict(X_t) for m in self.models])
        return self._ensemble.combine(preds)

    def predict(
        self,
        X: pd.DataFrame,
        eras: pd.Series | None = None,
    ) -> np.ndarray:
        """Generate predictions for new data.

        Args:
            X: Feature DataFrame (same schema as training data).
            eras: Optional era labels for per-era feature neutralization.
                When *None* and neutralization is enabled, neutralization is
                applied across the entire batch.

        Returns:
            1-D array of predictions. Rank-normalized when neutralization
            is active.
        """
        X_orig = X

        for pp in self.preprocessors:
            if isinstance(pp, TrainEvalPreprocessor):
                pp.eval()

        input_invalid_mask = invalid_row_mask(X)

        if input_invalid_mask.all():
            return np.full(len(X), 0.0)

        if input_invalid_mask.any():
            valid_mask = ~input_invalid_mask
            X_valid = X[valid_mask]
        else:
            valid_mask = None
            X_valid = X

        era_meta = protected_metadata_frame(X_valid)
        X_t = self._preprocess(X_valid, era_meta)

        post_nan_mask = ~X_t.isna().any(axis=1) if X_t.isna().any().any() else None
        if post_nan_mask is not None:
            X_t = X_t[post_nan_mask]

        if len(X_t) == 0:
            return np.full(len(X), 0.0)

        valid_preds = self._infer(X_t)

        if valid_mask is None and post_nan_mask is None:
            raw_preds = valid_preds
        else:
            raw_preds = np.full(len(X), safe_median(valid_preds))
            if valid_mask is not None and post_nan_mask is not None:
                combined = valid_mask.copy()
                valid_indices = np.where(valid_mask)[0]
                combined[valid_indices[~post_nan_mask]] = False
                raw_preds[combined] = valid_preds
            elif valid_mask is not None:
                raw_preds[valid_mask] = valid_preds
            else:
                raw_preds[post_nan_mask] = valid_preds

        if self._neutralizer is None:
            return raw_preds

        neutral_cols = (
            self.neutralize_features or self.feature_columns or list(X_orig.columns)
        )
        available_cols = [c for c in neutral_cols if c in X_orig.columns]
        if not available_cols:
            return rank_normalize(raw_preds)

        neutralized = self._neutralizer.neutralize(
            raw_preds,
            X_orig[available_cols],
            eras=eras,
        )
        return rank_normalize(neutralized)

    def to_numerai_predict(
        self,
        benchmark_col: str | None = None,
    ) -> Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """Create a Numerai-compatible ``predict`` callable.

        The returned function accepts ``(live_features, live_benchmark_models)``
        and returns a single-column ``prediction`` DataFrame with rank-normalized
        outputs in [0, 1].

        Args:
            benchmark_col: Optional benchmark model column to blend into
                the final prediction using ``benchmark_blend_weight``.

        Returns:
            A callable suitable for ``cloudpickle.dump`` and Numerai submission.
        """
        pipeline = self
        feature_columns = self.feature_columns or []
        blend_weight = self.benchmark_blend_weight
        bench_col = benchmark_col
        use_neutralization = pipeline._neutralizer is not None

        def predict(
            live_features: pd.DataFrame,
            live_benchmark_models: pd.DataFrame,
        ) -> pd.DataFrame:
            X = live_features[feature_columns]
            eras = (
                live_features["era"]
                if use_neutralization and "era" in live_features.columns
                else None
            )
            raw_preds = pipeline.predict(X, eras=eras)
            ranked = rank_normalize(raw_preds)

            if (
                blend_weight > 0.0
                and bench_col
                and bench_col in live_benchmark_models.columns
            ):
                bench = live_benchmark_models[bench_col].values
                ranked = (1.0 - blend_weight) * ranked + blend_weight * bench

            return pd.DataFrame({"prediction": ranked}, index=live_features.index)

        return predict

    def save_pipeline(self, path: Path) -> None:
        """Serialize the entire pipeline to disk via ``cloudpickle``.

        Args:
            path: Destination file path (parent directories are created
                automatically).
        """
        import cloudpickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load_pipeline(cls, path: Path) -> Self:
        """Load a previously saved pipeline from disk.

        Args:
            path: Path to the pickled pipeline file.

        Returns:
            The deserialized ``Pipeline`` instance.

        Raises:
            TypeError: If the loaded object is not a ``Pipeline``.
        """
        import cloudpickle

        with open(path, "rb") as f:
            loaded = cloudpickle.load(f)
            if not isinstance(loaded, cls):
                raise TypeError(
                    f"Expected a {cls.__name__} object, got {type(loaded).__name__}"
                )
            return loaded
