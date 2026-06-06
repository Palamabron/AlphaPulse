from pathlib import Path

import numpy as np
import pandas as pd


def smoke_test_predict_fn(
    pkl_path: Path,
    feature_columns: list[str],
    n_rows: int = 10,
) -> None:
    """Validate that a serialized predict.pkl works end-to-end.

    Loads the pickle, constructs synthetic input DataFrames (with correct
    features plus unexpected extra columns), calls the function, and asserts
    the output has the correct shape and values in [0, 1].

    Args:
        pkl_path: Path to the cloudpickle-serialized predict function.
        feature_columns: Expected feature column names used during training.
        n_rows: Number of synthetic rows to test with.

    Raises:
        RuntimeError: If the predict function cannot be loaded or produces
            invalid output (wrong type, missing column, wrong length, or
            out-of-range values).
    """
    import cloudpickle

    try:
        with open(pkl_path, "rb") as f:
            predict_fn = cloudpickle.load(f)
    except Exception as exc:
        raise RuntimeError(f"smoke_test: failed to load {pkl_path}: {exc}") from exc

    extra_cols = ["__smoke_extra_1__", "__smoke_extra_2__"]
    all_cols = list(feature_columns) + extra_cols
    rng = np.random.default_rng(0)
    live_features = pd.DataFrame(rng.random((n_rows, len(all_cols))), columns=all_cols)
    live_benchmark_models = pd.DataFrame(
        {"v2_equivalent_return": rng.random(n_rows)},
    )

    try:
        result = predict_fn(live_features, live_benchmark_models)
    except Exception as exc:
        raise RuntimeError(
            f"smoke_test: predict_fn raised on synthetic input: {exc}"
        ) from exc

    if not isinstance(result, pd.DataFrame):
        raise RuntimeError(
            f"smoke_test: expected DataFrame output, got {type(result).__name__}"
        )
    if "prediction" not in result.columns:
        raise RuntimeError(
            "smoke_test: output missing 'prediction' column. "
            f"Got: {list(result.columns)}"
        )
    if len(result) != n_rows:
        raise RuntimeError(f"smoke_test: expected {n_rows} rows, got {len(result)}")
    preds = result["prediction"].to_numpy(dtype=np.float64)
    finite = np.isfinite(preds)
    if not finite.all():
        raise RuntimeError(
            f"smoke_test: {(~finite).sum()} non-finite prediction value(s)"
        )
    if (preds < 0.0).any() or (preds > 1.0).any():
        raise RuntimeError(
            "smoke_test: predictions out of [0, 1]: "
            f"min={preds.min():.4f}, max={preds.max():.4f}"
        )
