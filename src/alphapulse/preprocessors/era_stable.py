from typing import Self

import numpy as np
import pandas as pd

from .base import BasePreprocessor, _PROTECTED_COLS

_MIN_ERAS_REQUIRED = 2


class EraStableFeatureSelector(BasePreprocessor):
    """Select features that are *consistently* important across eras.

    Trains a lightweight LightGBM model within each era (or a subsample of
    eras for speed), records per-feature importances, and ranks features by a
    blended score:

        score = (stability_weight * rank(stability)
            + (1 - stability_weight) * rank(mean_importance))

    where stability = mean_importance / std_importance across eras (higher is
    better — the feature is reliably important rather than spuriously so in a
    few eras).

    This targets a core Numerai pitfall: features with high *mean* importance
    but high *variance* across eras tend to overfit to specific market regimes.

    Args:
        keep_fraction: Fraction of features to keep (0, 1].
        n_estimators: Number of LightGBM trees per era model.
        stability_weight: Blend weight for stability vs mean importance (0–1).
            1.0 = select entirely by stability; 0.0 = select by mean importance.
        min_eras: Minimum number of eras required to compute stability.
            If fewer eras are available, falls back to mean importance only.
        max_era_subsample: Maximum number of eras to train on. Eras are sampled
            randomly when there are more than this many. None = use all eras.
        random_state: Seed for era subsampling and LightGBM.
    """

    def __init__(
        self,
        keep_fraction: float = 0.5,
        n_estimators: int = 50,
        stability_weight: float = 0.5,
        min_eras: int = 10,
        max_era_subsample: int | None = 50,
        random_state: int = 42,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
        if not 0.0 <= stability_weight <= 1.0:
            raise ValueError(
                f"stability_weight must be in [0, 1], got {stability_weight}"
            )
        self.keep_fraction = keep_fraction
        self.n_estimators = n_estimators
        self.stability_weight = stability_weight
        self.min_eras = min_eras
        self.max_era_subsample = max_era_subsample
        self.random_state = random_state
        self.selected_columns_: list[str] = []
        self.stability_scores_: pd.Series | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        eras: pd.Series | None = None,
    ) -> Self:
        if y is None:
            raise ValueError("EraStableFeatureSelector requires y for fit().")

        import lightgbm as lgb

        feature_cols = [c for c in X.columns if c not in _PROTECTED_COLS]
        n_features = len(feature_cols)
        n_keep = max(1, int(n_features * self.keep_fraction))

        if eras is None or len(pd.unique(eras)) < _MIN_ERAS_REQUIRED:
            # No era info — fall back to single global importance
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=1,
            )
            model.fit(X[feature_cols], y)
            importances = np.asarray(model.feature_importances_, dtype=np.float64)
            ranked = np.argsort(importances)[::-1][:n_keep]
            self.selected_columns_ = [str(feature_cols[i]) for i in ranked]
            self.stability_scores_ = pd.Series(importances, index=feature_cols)
            self.is_fitted = True
            return self

        era_arr = np.asarray(eras.to_numpy())
        unique_eras = sorted(pd.unique(era_arr), key=str)

        if (
            self.max_era_subsample is not None
            and len(unique_eras) > self.max_era_subsample
        ):
            rng = np.random.default_rng(self.random_state)
            unique_eras = list(
                rng.choice(unique_eras, size=self.max_era_subsample, replace=False)
            )

        era_importances: list[np.ndarray] = []
        for era in unique_eras:
            mask = era_arr == era
            X_era = X[mask][feature_cols]
            y_era = y[mask]
            if len(X_era) < 20:
                continue
            if float(y_era.std()) == 0.0:
                continue
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=1,
            )
            try:
                model.fit(X_era, y_era)
            except Exception:  # noqa: S112
                continue
            era_importances.append(
                np.asarray(model.feature_importances_, dtype=np.float64)
            )

        if len(era_importances) < max(self.min_eras, _MIN_ERAS_REQUIRED):
            # Not enough eras — use global importance
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=1,
            )
            model.fit(X[feature_cols], y)
            importances = np.asarray(model.feature_importances_, dtype=np.float64)
            ranked = np.argsort(importances)[::-1][:n_keep]
            self.selected_columns_ = [str(feature_cols[i]) for i in ranked]
            self.stability_scores_ = pd.Series(importances, index=feature_cols)
            self.is_fitted = True
            return self

        imp_matrix = np.stack(era_importances, axis=0)  # shape: (n_eras, n_features)
        mean_imp = imp_matrix.mean(axis=0)
        std_imp = imp_matrix.std(axis=0, ddof=0)

        # stability = mean / std (higher = more consistent)
        # Add small epsilon to avoid division by zero for always-zero features
        stability = np.where(std_imp > 0, mean_imp / (std_imp + 1e-10), mean_imp)

        rank_stability = _rank_asc(stability)
        rank_mean = _rank_asc(mean_imp)
        blended = (
            self.stability_weight * rank_stability
            + (1.0 - self.stability_weight) * rank_mean
        )

        top_indices = np.argsort(blended)[::-1][:n_keep]
        self.selected_columns_ = [str(feature_cols[i]) for i in top_indices]
        self.stability_scores_ = pd.Series(
            blended, index=feature_cols, name="stability_score"
        )

        self.is_fitted = True
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        *,
        eras: pd.Series | None = None,
    ) -> pd.DataFrame:
        self.fit(X, y, eras=eras)
        return self.transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("EraStableFeatureSelector not fitted.")
        cols = [c for c in self.selected_columns_ if c in X.columns]
        return X[cols].copy()


def _rank_asc(arr: np.ndarray) -> np.ndarray:
    """Return 0-based ascending ranks (ties broken by first occurrence)."""
    order = np.argsort(arr, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(arr))
    return ranks.astype(np.float64)
