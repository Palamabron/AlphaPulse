from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..constants import _PROTECTED_COLS
from ..evaluation.metrics import (
    era_sharpe,
    era_sharpe_of_mmc,
    payout_score,
    rank_normalize,
)


def _numeric_features(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(features, pd.DataFrame):
        feat_cols = [c for c in features.columns if c not in _PROTECTED_COLS]
        numeric = features[feat_cols].select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError("FeatureNeutralizer: no numeric feature columns found.")
        return np.asarray(numeric.values, dtype=np.float64)
    return np.asarray(features, dtype=np.float64)


def _aligned_vector(
    values: np.ndarray | pd.Series,
    index: pd.Index | None,
    *,
    name: str,
    dtype: type[np.float64] | None = np.float64,
) -> np.ndarray:
    if isinstance(values, pd.Series) and index is not None:
        if values.index.has_duplicates:
            raise ValueError(f"{name} index must not contain duplicates")
        missing = index.difference(values.index)
        extra = values.index.difference(index)
        if len(missing) > 0 or len(extra) > 0:
            raise ValueError(f"{name} row IDs must exactly match feature row IDs")
        return np.asarray(values.reindex(index).to_numpy(dtype=dtype), dtype=dtype)
    array = np.asarray(values, dtype=dtype)
    if index is not None and len(array) != len(index):
        raise ValueError(
            f"{name} length {len(array)} != feature row count {len(index)}"
        )
    return array


def neutralize_against_meta(
    predictions: np.ndarray | pd.Series,
    meta_model: np.ndarray | pd.Series,
    eras: pd.Series | None = None,
    proportion: float = 0.5,
) -> np.ndarray:
    """Remove linear exposure of predictions to the Numerai meta model (per era)."""
    if proportion == 0.0:
        return np.asarray(predictions, dtype=np.float64)
    reference_index = predictions.index if isinstance(predictions, pd.Series) else None
    pred_arr = _aligned_vector(
        predictions, reference_index, name="predictions"
    ).reshape(-1)
    meta_arr = _aligned_vector(meta_model, reference_index, name="meta_model").reshape(
        -1
    )
    if len(pred_arr) != len(meta_arr):
        raise ValueError(
            f"predictions length {len(pred_arr)} != meta_model length {len(meta_arr)}"
        )

    def _neutralize_slice(pred: np.ndarray, meta: np.ndarray) -> np.ndarray:
        if len(pred) < 2:
            return pred.copy()
        meta_c = meta - meta.mean()
        pred_c = pred - pred.mean()
        denom = float(np.dot(meta_c, meta_c))
        if denom <= 1e-12:
            return pred.copy()
        beta = float(np.dot(meta_c, pred_c) / denom)
        exposure = meta_c * beta
        return np.asarray(pred - proportion * exposure, dtype=np.float64)

    if eras is None:
        return _neutralize_slice(pred_arr, meta_arr)

    era_vals = _aligned_vector(eras, reference_index, name="eras", dtype=None).reshape(
        -1
    )
    if len(era_vals) != len(pred_arr):
        raise ValueError(
            f"eras length {len(era_vals)} != predictions length {len(pred_arr)}"
        )
    out = pred_arr.copy()
    for era in np.unique(era_vals):
        mask = era_vals == era
        if mask.sum() < 2:
            continue
        out[mask] = _neutralize_slice(pred_arr[mask], meta_arr[mask])
    return out


class MetaModelNeutralizer:
    def __init__(self, proportion: float = 0.5) -> None:
        if not 0.0 <= proportion <= 1.0:
            raise ValueError(f"proportion must be in [0, 1], got {proportion}")
        self.proportion = proportion

    def neutralize(
        self,
        predictions: np.ndarray | pd.Series,
        meta_model: np.ndarray | pd.Series,
        eras: pd.Series | None = None,
    ) -> np.ndarray:
        return neutralize_against_meta(
            predictions, meta_model, eras=eras, proportion=self.proportion
        )

    def optimize_proportion(
        self,
        predictions: np.ndarray | pd.Series,
        meta_model: np.ndarray | pd.Series,
        y_true: pd.Series,
        eras: pd.Series,
        *,
        objective: Literal["corr_sharpe", "mmc_sharpe", "payout_score"] = "mmc_sharpe",
        bounds: tuple[float, float] = (0.0, 1.0),
        corr_weight: float = 0.75,
        mmc_weight: float = 2.25,
    ) -> float:
        pred_arr = np.asarray(predictions, dtype=np.float64)
        meta_arr = np.asarray(meta_model, dtype=np.float64)

        def neg_score(p: float) -> float:
            out = neutralize_against_meta(pred_arr, meta_arr, eras=eras, proportion=p)
            if objective == "payout_score":
                score = payout_score(
                    y_true, out, meta_arr, eras, corr_weight, mmc_weight
                )
            elif objective == "mmc_sharpe":
                score = era_sharpe_of_mmc(y_true, out, meta_arr, eras)
            else:
                score = era_sharpe(y_true, out, eras)
            return -score if np.isfinite(score) else 1e6

        result = minimize_scalar(
            neg_score,
            bounds=bounds,
            method="bounded",
            options={"maxiter": 200, "xatol": 1e-4},
        )
        self.proportion = float(result.x)
        return self.proportion


class FeatureNeutralizer:
    def __init__(self, proportion: float = 0.5) -> None:
        if not 0.0 <= proportion <= 1.0:
            raise ValueError(f"proportion must be in [0, 1], got {proportion}")
        self.proportion = proportion

    @staticmethod
    def _neutralize_array(
        predictions: np.ndarray,
        features: np.ndarray,
        proportion: float,
    ) -> np.ndarray:
        if proportion == 0.0:
            return predictions.copy()

        feat = features - features.mean(axis=0)
        pred = predictions - predictions.mean()

        try:
            beta, _, _, _ = np.linalg.lstsq(feat, pred, rcond=None)
        except np.linalg.LinAlgError:
            return predictions.copy()

        exposure = feat @ beta
        return np.asarray(
            pred - proportion * exposure + predictions.mean(), dtype=np.float64
        )

    def neutralize(
        self,
        predictions: np.ndarray | pd.Series,
        features: pd.DataFrame | np.ndarray,
        eras: pd.Series | None = None,
    ) -> np.ndarray:
        reference_index = features.index if isinstance(features, pd.DataFrame) else None
        pred_arr = _aligned_vector(
            predictions, reference_index, name="predictions"
        ).reshape(-1)
        feat_arr = _numeric_features(features)

        if eras is None:
            return self._neutralize_array(pred_arr, feat_arr, self.proportion)

        era_vals = _aligned_vector(
            eras, reference_index, name="eras", dtype=None
        ).reshape(-1)
        if len(era_vals) != len(pred_arr):
            raise ValueError(
                f"eras length {len(era_vals)} != predictions length {len(pred_arr)}"
            )
        out = pred_arr.copy()
        for era in np.unique(era_vals):
            mask = era_vals == era
            if mask.sum() < 2:
                continue
            out[mask] = self._neutralize_array(
                pred_arr[mask], feat_arr[mask], self.proportion
            )
        return out

    def optimize_proportion(
        self,
        predictions: np.ndarray | pd.Series,
        features: pd.DataFrame | np.ndarray,
        y_true: pd.Series,
        eras: pd.Series,
        bounds: tuple[float, float] = (0.0, 1.0),
        *,
        objective: Literal["corr_sharpe", "mmc_sharpe", "payout_score"] = "corr_sharpe",
        meta_model: np.ndarray | pd.Series | None = None,
        corr_weight: float = 0.75,
        mmc_weight: float = 2.25,
    ) -> float:
        pred_arr = np.asarray(predictions, dtype=np.float64)
        feat_arr = _numeric_features(features)
        era_vals = np.asarray(eras)
        meta_arr = (
            np.asarray(meta_model, dtype=np.float64) if meta_model is not None else None
        )

        def neg_score(p: float) -> float:
            out = pred_arr.copy()
            for era in np.unique(era_vals):
                mask = era_vals == era
                if mask.sum() < 2:
                    continue
                out[mask] = self._neutralize_array(pred_arr[mask], feat_arr[mask], p)
            if objective == "payout_score" and meta_arr is not None:
                score = payout_score(
                    y_true, out, meta_arr, eras, corr_weight, mmc_weight
                )
            elif objective == "mmc_sharpe" and meta_arr is not None:
                score = era_sharpe_of_mmc(y_true, out, meta_arr, eras)
            else:
                score = era_sharpe(y_true, out, eras)
            return -score if np.isfinite(score) else 1e6

        result = minimize_scalar(
            neg_score,
            bounds=bounds,
            method="bounded",
            options={"maxiter": 200, "xatol": 1e-4},
        )
        self.proportion = float(result.x)
        return self.proportion


def apply_prediction_neutralization(
    predictions: np.ndarray | pd.Series,
    features: pd.DataFrame,
    *,
    eras: pd.Series | None = None,
    feature_columns: list[str] | None = None,
    neutralize_features: list[str] | None = None,
    feature_neutralizer: FeatureNeutralizer | None = None,
    meta_model: np.ndarray | pd.Series | None = None,
    meta_neutralizer: MetaModelNeutralizer | None = None,
) -> np.ndarray:
    output = _aligned_vector(predictions, features.index, name="predictions").reshape(
        -1
    )
    aligned_eras = (
        pd.Series(
            _aligned_vector(eras, features.index, name="eras", dtype=None),
            index=features.index,
        )
        if eras is not None
        else None
    )
    if feature_neutralizer is not None:
        if neutralize_features is not None:
            missing = [
                column
                for column in neutralize_features
                if column not in features.columns
            ]
            if missing:
                raise ValueError(
                    "Prediction neutralization is missing configured feature "
                    f"columns: {missing[:10]}"
                )
        requested = neutralize_features or feature_columns or list(features.columns)
        available = [column for column in requested if column in features.columns]
        if available:
            output = feature_neutralizer.neutralize(
                output,
                features[available],
                eras=aligned_eras,
            )
        output = rank_normalize(output)

    if meta_neutralizer is None:
        return output
    if meta_model is None:
        raise ValueError(
            "Meta-model neutralization is configured, but aligned meta-model "
            "predictions were not provided"
        )

    meta_arr = _aligned_vector(meta_model, features.index, name="meta_model").reshape(
        -1
    )
    if len(meta_arr) != len(output):
        raise ValueError(
            f"meta_model length {len(meta_arr)} != predictions length {len(output)}"
        )
    if not np.isfinite(meta_arr).all():
        raise ValueError("meta_model contains missing or non-finite predictions")
    output = meta_neutralizer.neutralize(output, meta_arr, eras=aligned_eras)
    return rank_normalize(output)
