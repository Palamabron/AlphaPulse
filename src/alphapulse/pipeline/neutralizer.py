import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..constants import _PROTECTED_COLS
from ..evaluation.metrics import era_sharpe


def _numeric_features(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(features, pd.DataFrame):
        feat_cols = [c for c in features.columns if c not in _PROTECTED_COLS]
        numeric = features[feat_cols].select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError("FeatureNeutralizer: no numeric feature columns found.")
        return np.asarray(numeric.values, dtype=np.float64)
    return np.asarray(features, dtype=np.float64)


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
        pred_arr = np.asarray(predictions, dtype=np.float64)
        feat_arr = _numeric_features(features)

        if eras is None:
            return self._neutralize_array(pred_arr, feat_arr, self.proportion)

        era_vals = np.asarray(eras)
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
    ) -> float:
        pred_arr = np.asarray(predictions, dtype=np.float64)
        feat_arr = _numeric_features(features)
        era_vals = np.asarray(eras)

        def neg_sharpe(p: float) -> float:
            out = pred_arr.copy()
            for era in np.unique(era_vals):
                mask = era_vals == era
                if mask.sum() < 2:
                    continue
                out[mask] = self._neutralize_array(pred_arr[mask], feat_arr[mask], p)
            s = era_sharpe(y_true, out, eras)
            return -s if np.isfinite(s) else 1e6

        result = minimize_scalar(
            neg_sharpe,
            bounds=bounds,
            method="bounded",
            options={"maxiter": 200, "xatol": 1e-4},
        )
        self.proportion = float(result.x)
        return self.proportion
