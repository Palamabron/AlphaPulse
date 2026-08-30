from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from ..validation.purged_cv import PurgedEraCV
from .backtester import Backtester, PredictorProtocol, predict_with_optional_eras
from .metrics import calculate_metrics, rank_normalize_per_era

if TYPE_CHECKING:
    from ..pipeline.neutralizer import FeatureNeutralizer

WF_N_SPLITS = 3
WF_N_PURGE = 8
WF_MIN_TRAIN_ERAS = 20

HPO_FAST_HOLDOUT_ERAS = 52
HPO_FAST_WF_N_SPLITS = 2
HPO_FAST_MAX_TRAIN_ERAS = 120
HPO_FAST_N_SUBS_CAP = 5

_NAN_METRICS: dict[str, float] = {
    "mean_per_era_correlation": float("nan"),
    "std_per_era_correlation": float("nan"),
    "corr_sharpe": float("nan"),
    "max_drawdown": float("nan"),
    "pct_positive_eras": float("nan"),
    "n_valid_eras": float("nan"),
}


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
    eras_sorted = sorted(era_val.unique(), key=str)
    holdout_eras = eras_sorted[-last_n_eras:]
    mask = era_val.isin(holdout_eras)
    if not mask.any():
        return {
            "mean_per_era_correlation": float("nan"),
            "std_per_era_correlation": float("nan"),
            "corr_sharpe": float("nan"),
        }
    bt = Backtester(predictor, feature_columns=feature_columns)
    return bt.evaluate(X_val.loc[mask], y_val.loc[mask], era_val.loc[mask])


class EraSplitEvaluator:
    """Walk-forward backtester with optional purge/embargo gaps.

    Two splitting strategies are available depending on whether *n_splits* is
    set:

    - **Expanding window** (``n_splits=None``): tests one era at a time.
      When ``n_purge > 0``, the *n_purge* eras immediately preceding each test
      era are excluded from training to prevent look-ahead bias (important for
      Numerai where target windows overlap across adjacent eras).  When
      ``n_embargo > 0``, consecutive test eras are separated by *n_embargo*
      skipped eras, so the loop steps by ``n_embargo + 1``.

    - **PurgedEraCV** (``n_splits >= 2``): delegates fold generation to
      :class:`~alphapulse.validation.purged_cv.PurgedEraCV`, producing
      *n_splits* folds each containing multiple test eras.  Faster than the
      expanding window but less granular.

    In both modes the returned metrics dict matches the full output of
    :func:`~alphapulse.evaluation.metrics.calculate_metrics`.
    """

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        min_train_eras: int = 1,
        n_purge: int = 8,
        n_embargo: int = 0,
        n_splits: int | None = None,
        max_train_eras: int | None = None,
        neutralizer: "FeatureNeutralizer | None" = None,
    ) -> None:
        if n_purge < 0:
            raise ValueError("n_purge must be >= 0")
        if n_embargo < 0:
            raise ValueError("n_embargo must be >= 0")
        if n_splits is not None and n_splits < 2:
            raise ValueError("n_splits must be None or >= 2")
        self.feature_columns = feature_columns
        self.min_train_eras = min_train_eras
        self.n_purge = n_purge
        self.n_embargo = n_embargo
        self.n_splits = n_splits
        self.max_train_eras = max_train_eras
        self.neutralizer = neutralizer

    def evaluate_walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        era: pd.Series,
        train_fn: Callable[[pd.DataFrame, pd.Series], PredictorProtocol],
        eras_order: list[Any] | None = None,
        meta_model: pd.Series | None = None,
        last_fold_callback: Callable[
            [PredictorProtocol, pd.DataFrame, pd.Series, pd.Series], None
        ]
        | None = None,
    ) -> dict[str, float]:
        X_use = X[self.feature_columns] if self.feature_columns is not None else X
        df = pd.concat(
            [X_use, pd.DataFrame({"y": y, "era": era}, index=X.index)], axis=1
        )

        if eras_order is None:
            eras_order = sorted(df["era"].unique(), key=str)

        all_y_true: list[float] = []
        all_y_pred: list[float] = []
        all_era: list[Any] = []
        all_meta: list[float] = []

        if self.n_splits is not None:
            self._collect_purged_cv(
                df,
                X_use,
                train_fn,
                all_y_true,
                all_y_pred,
                all_era,
                meta_model=meta_model,
                all_meta=all_meta if meta_model is not None else None,
                last_fold_callback=last_fold_callback,
            )
        else:
            self._collect_expanding(
                df,
                X_use,
                eras_order,
                train_fn,
                all_y_true,
                all_y_pred,
                all_era,
                meta_model=meta_model,
                all_meta=all_meta if meta_model is not None else None,
                last_fold_callback=last_fold_callback,
            )

        if not all_y_true:
            return dict(_NAN_METRICS)

        y_s = pd.Series(all_y_true)
        era_s = pd.Series(all_era)
        pred_a = rank_normalize_per_era(np.asarray(all_y_pred, dtype=np.float64), era_s)
        meta_arr = np.asarray(all_meta, dtype=np.float64) if all_meta else None
        return calculate_metrics(y_s, pred_a, era_s, meta_model_preds=meta_arr)

    def _collect_expanding(
        self,
        df: pd.DataFrame,
        X_use: pd.DataFrame,
        eras_order: list[Any],
        train_fn: Callable[[pd.DataFrame, pd.Series], PredictorProtocol],
        all_y_true: list[float],
        all_y_pred: list[float],
        all_era: list[Any],
        *,
        meta_model: pd.Series | None = None,
        all_meta: list[float] | None = None,
        last_fold_callback: Callable[
            [PredictorProtocol, pd.DataFrame, pd.Series, pd.Series], None
        ]
        | None = None,
    ) -> None:
        step = self.n_embargo + 1
        first_test = self.min_train_eras + self.n_purge
        for i in range(first_test, len(eras_order), step):
            train_end = max(0, i - self.n_purge)
            train_eras = eras_order[:train_end]
            if len(train_eras) < self.min_train_eras:
                continue
            test_era = eras_order[i]
            train_mask = df["era"].isin(set(train_eras))
            test_mask = df["era"] == test_era
            if not test_mask.any():
                continue
            train_cols = list(X_use.columns)
            if "era" in df.columns and "era" not in X_use.columns:
                train_cols.append("era")
            X_tr = cast(pd.DataFrame, df.loc[train_mask, train_cols])
            y_tr = cast(pd.Series, df.loc[train_mask, "y"])
            X_te = cast(pd.DataFrame, df.loc[test_mask, X_use.columns])
            y_te = cast(pd.Series, df.loc[test_mask, "y"])
            predictor = train_fn(X_tr, y_tr)
            era_labels = pd.Series([test_era] * len(y_te), index=y_te.index)
            preds = predict_with_optional_eras(predictor, X_te, era_labels)
            if self.neutralizer is not None:
                preds = self.neutralizer.neutralize(preds, X_te, era_labels)
            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(np.asarray(preds).ravel().tolist())
            all_era.extend([test_era] * len(y_te))
            if meta_model is not None and all_meta is not None:
                all_meta.extend(meta_model.loc[y_te.index].tolist())
            if last_fold_callback is not None:
                last_fold_callback(predictor, X_te, y_te, era_labels)

    def _collect_purged_cv(
        self,
        df: pd.DataFrame,
        X_use: pd.DataFrame,
        train_fn: Callable[[pd.DataFrame, pd.Series], PredictorProtocol],
        all_y_true: list[float],
        all_y_pred: list[float],
        all_era: list[Any],
        *,
        meta_model: pd.Series | None = None,
        all_meta: list[float] | None = None,
        last_fold_callback: Callable[
            [PredictorProtocol, pd.DataFrame, pd.Series, pd.Series], None
        ]
        | None = None,
    ) -> None:
        era_series = df["era"]
        cv = PurgedEraCV(
            n_splits=self.n_splits,  # type: ignore[arg-type]
            n_purge=self.n_purge,
            n_embargo=self.n_embargo,
            min_train_eras=self.min_train_eras,
            max_train_eras=self.max_train_eras,
        )
        for train_eras, test_eras in cv.split_eras(era_series):
            train_mask = era_series.isin(set(train_eras))
            test_mask = era_series.isin(set(test_eras))
            if not test_mask.any():
                continue
            train_cols = list(X_use.columns)
            if "era" in df.columns and "era" not in X_use.columns:
                train_cols.append("era")
            X_tr = cast(pd.DataFrame, df.loc[train_mask, train_cols])
            y_tr = cast(pd.Series, df.loc[train_mask, "y"])
            X_te = cast(pd.DataFrame, df.loc[test_mask, X_use.columns])
            y_te = cast(pd.Series, df.loc[test_mask, "y"])
            predictor = train_fn(X_tr, y_tr)
            fold_eras = era_series.loc[test_mask]
            preds = predict_with_optional_eras(predictor, X_te, fold_eras)
            if self.neutralizer is not None:
                preds = self.neutralizer.neutralize(preds, X_te, fold_eras)
            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(np.asarray(preds).ravel().tolist())
            all_era.extend(era_series.loc[test_mask].tolist())
            if meta_model is not None and all_meta is not None:
                all_meta.extend(meta_model.loc[y_te.index].tolist())
            if last_fold_callback is not None:
                last_fold_callback(predictor, X_te, y_te, fold_eras)


__all__ = [
    "HPO_FAST_HOLDOUT_ERAS",
    "HPO_FAST_MAX_TRAIN_ERAS",
    "HPO_FAST_N_SUBS_CAP",
    "HPO_FAST_WF_N_SPLITS",
    "WF_MIN_TRAIN_ERAS",
    "WF_N_PURGE",
    "WF_N_SPLITS",
    "EraSplitEvaluator",
    "PredictorProtocol",
    "evaluate_holdout_last_n_eras",
]
