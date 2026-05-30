"""Per-era feature importance diagnostics for Numerai pipelines.

Computes feature importances within each era and tracks their stability.
Stable features (consistently important across market regimes) are better
candidates for Numerai models than features that are sporadically important.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_feature_report(
    X: pd.DataFrame,
    y: pd.Series,
    eras: pd.Series,
    *,
    n_estimators: int = 50,
    top_n: int = 20,
    max_era_subsample: int | None = 50,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute per-era feature importance report for gradient boosting models.

    Trains a small LightGBM model within each era, records feature importances,
    and summarizes stability (mean/std across eras).

    Args:
        X: Feature DataFrame (training or validation set).
        y: Target Series aligned with X.
        eras: Era labels aligned with X.
        n_estimators: Number of LightGBM trees per era model.
        top_n: Number of features to include in top/bottom lists.
        max_era_subsample: Cap on number of eras to use (randomly sampled).
            None = use all eras.
        random_state: Seed for era subsampling and LightGBM.

    Returns:
        Dict with keys:
          - ``top_by_mean``: list of {feature, mean_importance} dicts.
          - ``top_by_stability``: list of {feature, stability, mean_importance}.
          - ``bottom_by_stability``: same format, lowest stability.
          - ``n_eras_used``: number of era models trained.
          - ``n_features``: total feature count.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required for feature reports. "
            "Install with: uv sync --extra dev"
        ) from exc

    feature_cols = list(X.columns)
    era_arr = np.asarray(eras.to_numpy())
    unique_eras = sorted(pd.unique(era_arr), key=str)

    if max_era_subsample is not None and len(unique_eras) > max_era_subsample:
        rng = np.random.default_rng(random_state)
        unique_eras = list(
            rng.choice(unique_eras, size=max_era_subsample, replace=False)
        )

    era_importances: list[np.ndarray] = []
    for era in unique_eras:
        mask = era_arr == era
        X_era = X[mask]
        y_era = y[mask]
        if len(X_era) < 20 or float(y_era.std()) == 0.0:
            continue
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state,
            verbosity=-1,
            n_jobs=1,
        )
        try:
            model.fit(X_era, y_era)
        except Exception:  # noqa: S112
            continue
        era_importances.append(np.asarray(model.feature_importances_, dtype=np.float64))

    n_eras = len(era_importances)
    if n_eras == 0:
        return {
            "top_by_mean": [],
            "top_by_stability": [],
            "bottom_by_stability": [],
            "n_eras_used": 0,
            "n_features": len(feature_cols),
        }

    imp_matrix = np.stack(era_importances, axis=0)
    mean_imp = imp_matrix.mean(axis=0)
    std_imp = imp_matrix.std(axis=0, ddof=0)
    stability = np.where(std_imp > 0, mean_imp / (std_imp + 1e-10), mean_imp)

    summary = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_importance": mean_imp,
            "std_importance": std_imp,
            "stability": stability,
        }
    ).set_index("feature")

    top_mean = summary.nlargest(top_n, "mean_importance")
    top_stab = summary.nlargest(top_n, "stability")
    bot_stab = summary.nsmallest(top_n, "stability")

    return {
        "top_by_mean": _to_records(top_mean, ["mean_importance"]),
        "top_by_stability": _to_records(top_stab, ["stability", "mean_importance"]),
        "bottom_by_stability": _to_records(bot_stab, ["stability", "mean_importance"]),
        "n_eras_used": n_eras,
        "n_features": len(feature_cols),
    }


def _to_records(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    rows = []
    for feat, row in df.iterrows():
        entry: dict[str, Any] = {"feature": str(feat)}
        for c in cols:
            val = float(row[c])
            entry[c] = round(val, 6)
        rows.append(entry)
    return rows
