from collections.abc import Callable
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from .backtester import Backtester
from .metrics import era_correlation_metrics, per_era_correlation


class PredictorProtocol(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def evaluate_holdout_last_n_eras(
    predictor: PredictorProtocol,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    era_val: pd.Series,
    feature_columns: list[str] | None,
    last_n_eras: int,
) -> dict[str, float]:
    if last_n_eras < 1:
        raise ValueError("last_n_eras must be >= 1")
    eras_sorted = sorted(era_val.unique(), key=lambda x: str(x))
    holdout_eras = eras_sorted[-last_n_eras:]
    mask = era_val.isin(holdout_eras)
    if not mask.any():
        return {
            "mean_per_era_correlation": float("nan"),
            "std_per_era_correlation": float("nan"),
            "sharpe": float("nan"),
            "correlation": float("nan"),
        }
    bt = Backtester(predictor, feature_columns=feature_columns)
    return bt.evaluate(X_val.loc[mask], y_val.loc[mask], era_val.loc[mask])


class EraSplitEvaluator:
    """Walk-forward: for each test era, fit on prior eras, predict test era."""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        min_train_eras: int = 1,
    ) -> None:
        self.feature_columns = feature_columns
        self.min_train_eras = min_train_eras

    def evaluate_walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series,
        train_fn: Callable[[pd.DataFrame, pd.Series], PredictorProtocol],
        eras_order: list[Any] | None = None,
    ) -> dict[str, float]:
        df = pd.DataFrame({"y": y, "era": era}, index=X.index)
        X_use = X[self.feature_columns] if self.feature_columns is not None else X
        df = pd.concat([X_use, df], axis=1)

        if eras_order is None:
            eras_order = sorted(df["era"].unique(), key=lambda x: str(x))
        if len(eras_order) < self.min_train_eras + 1:
            return {
                "mean_per_era_correlation": float("nan"),
                "std_per_era_correlation": float("nan"),
                "sharpe": float("nan"),
                "correlation": float("nan"),
            }

        all_y_true: list[float] = []
        all_y_pred: list[float] = []
        all_era: list[Any] = []

        for i, test_era in enumerate(eras_order):
            train_eras = eras_order[:i]
            if len(train_eras) < self.min_train_eras:
                continue
            train_mask = df["era"].isin(train_eras)
            test_mask = df["era"] == test_era
            if not test_mask.any():
                continue

            X_tr = cast(pd.DataFrame, df.loc[train_mask, X_use.columns])
            y_tr = cast(pd.Series, df.loc[train_mask, "y"])
            X_te = cast(pd.DataFrame, df.loc[test_mask, X_use.columns])
            y_te = cast(pd.Series, df.loc[test_mask, "y"])

            predictor = train_fn(X_tr, y_tr)
            preds = predictor.predict(X_te)
            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(np.asarray(preds).ravel().tolist())
            all_era.extend([test_era] * len(y_te))

        if not all_y_true:
            return {
                "mean_per_era_correlation": float("nan"),
                "std_per_era_correlation": float("nan"),
                "sharpe": float("nan"),
                "correlation": float("nan"),
            }

        y_s = pd.Series(all_y_true)
        era_s = pd.Series(all_era)
        pred_a = np.asarray(all_y_pred, dtype=np.float64)

        per_era = per_era_correlation(y_s, pred_a, era_s)
        valid = per_era.dropna()
        scoring = era_correlation_metrics(y_s, pred_a, era_s)

        return {
            "mean_per_era_correlation": float(valid.mean())
            if len(valid) > 0
            else float("nan"),
            "std_per_era_correlation": float(valid.std())
            if len(valid) > 1
            else float("nan"),
            "sharpe": scoring["corr_sharpe"],
            "correlation": scoring["mean_era_corr"],
        }
