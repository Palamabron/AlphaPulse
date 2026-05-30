"""Numerai submission format validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_submission(
    predictions_df: pd.DataFrame,
    live_df: pd.DataFrame | None = None,
    *,
    id_col: str = "id",
    pred_col: str = "prediction",
) -> list[str]:
    """Validate a Numerai predictions DataFrame for submission correctness.

    Args:
        predictions_df: DataFrame with predictions to validate. Must have
            at minimum a ``prediction`` column. The row index or an ``id``
            column is used for ID alignment checks.
        live_df: Optional live feature DataFrame for ID alignment checks.
            When provided, all live IDs must appear in predictions.
        id_col: Name of the ID column. If not present, the index is used.
        pred_col: Name of the prediction column.

    Returns:
        List of error/warning strings. Empty list = submission is valid.
    """
    issues: list[str] = []

    # Check prediction column exists
    if pred_col not in predictions_df.columns:
        issues.append(f"Missing required column '{pred_col}'.")
        return issues  # can't check further without predictions

    preds = predictions_df[pred_col]

    # Check for NaN predictions
    n_nan = int(preds.isna().sum())
    if n_nan > 0:
        issues.append(f"ERROR: {n_nan} NaN predictions found.")

    # Check predictions are in [0, 1]
    valid_preds = preds.dropna()
    if len(valid_preds) > 0:
        p_min = float(valid_preds.min())
        p_max = float(valid_preds.max())
        if p_min < 0.0 or p_max > 1.0:
            issues.append(
                f"ERROR: Predictions must be in [0, 1]. "
                f"Got min={p_min:.4f}, max={p_max:.4f}."
            )

    # Check for constant predictions (degenerate)
    if len(valid_preds) > 1 and float(valid_preds.std()) == 0.0:
        issues.append(
            "WARNING: All predictions are identical (constant). This will score 0 CORR."
        )

    # Check for very low variance (near-constant)
    if len(valid_preds) > 10:
        unique_ratio = valid_preds.nunique() / len(valid_preds)
        if unique_ratio < 0.01:
            issues.append(
                f"WARNING: Very low prediction diversity "
                f"({valid_preds.nunique()} unique values out of {len(valid_preds)}). "
                "Model may be degenerate."
            )

    # ID alignment check against live data
    if live_df is not None:
        live_ids = set(
            live_df[id_col].tolist()
            if id_col in live_df.columns
            else live_df.index.tolist()
        )
        pred_ids = set(
            predictions_df[id_col].tolist()
            if id_col in predictions_df.columns
            else predictions_df.index.tolist()
        )

        missing_ids = live_ids - pred_ids
        extra_ids = pred_ids - live_ids

        if missing_ids:
            issues.append(
                f"ERROR: {len(missing_ids)} live IDs are missing from predictions."
            )
        if extra_ids:
            issues.append(
                f"WARNING: {len(extra_ids)} prediction IDs are not in live data "
                "(extra rows will be ignored by Numerai)."
            )

    return issues


def prepare_submission(
    predictions: np.ndarray,
    ids: pd.Index | list[str],
    *,
    pred_col: str = "prediction",
) -> pd.DataFrame:
    """Create a Numerai-format submission DataFrame.

    Args:
        predictions: 1-D array of predictions (will be rank-normalized to [0,1]).
        ids: Row identifiers (from the live features DataFrame index or ``id`` column).
        pred_col: Name for the prediction column.

    Returns:
        DataFrame with ``id`` and ``prediction`` columns, predictions in [0, 1].
    """
    from .metrics import rank_normalize

    preds_norm = rank_normalize(np.asarray(predictions, dtype=np.float64))
    return pd.DataFrame({"id": list(ids), pred_col: preds_norm})
