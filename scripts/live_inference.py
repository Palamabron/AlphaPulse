"""Run a trained Numerai model on live data and produce a submission CSV.

Usage:
    uv run python scripts/live_inference.py \\
        --model-path artifacts/competition_pickle/predict.pkl \\
        --data-dir data/v5.2 \\
        --output-path artifacts/live/predictions.csv
"""

import sys
from pathlib import Path

import cloudpickle
import pandas as pd
import tyro
from loguru import logger

from alphapulse.evaluation.submission import prepare_submission, validate_submission


def main(
    model_path: Path,
    data_dir: Path = Path("data/v5.2"),
    output_path: Path = Path("artifacts/live/predictions.csv"),
    benchmark_col: str = "v2_equivalent_return",
    validate: bool = True,
) -> None:
    """Run live inference and produce a Numerai submission CSV.

    Args:
        model_path: Path to a cloudpickle-serialised ``predict`` callable
            (as produced by ``scripts/export_numerai_pickle.py``).
        data_dir: Numerai data directory containing ``live.parquet``.
        output_path: Path for the output CSV file.
        benchmark_col: Benchmark column name passed to the predict callable.
        validate: Run submission format validation before saving.
    """
    if not model_path.exists():
        logger.error("Model not found: {}", model_path)
        raise SystemExit(1)

    live_path = data_dir / "live.parquet"
    if not live_path.exists():
        logger.error(
            "Live data not found: {}. Run scripts/download_dataset.py first.", live_path
        )
        raise SystemExit(1)

    logger.info("Loading model from {}", model_path)
    with open(model_path, "rb") as f:
        predict_fn = cloudpickle.load(f)

    logger.info("Loading live data from {}", live_path)
    live_df = pd.read_parquet(live_path)
    logger.info("Live data: {} rows, {} columns", len(live_df), len(live_df.columns))

    logger.info("Running inference...")
    try:
        # Numerai-compatible callable: predict(live_features, live_benchmark_models)
        if benchmark_col in live_df.columns:
            benchmark = live_df[[benchmark_col]]
            predictions = predict_fn(live_df, benchmark)
        else:
            # Fall back: call with live_df only
            predictions = predict_fn(live_df, None)
    except Exception as exc:
        logger.error("Inference failed: {}", exc)
        raise SystemExit(1) from exc

    # Extract prediction series
    if isinstance(predictions, pd.DataFrame):
        pred_col = (
            "prediction"
            if "prediction" in predictions.columns
            else predictions.columns[0]
        )
        pred_series = predictions[pred_col]
    elif isinstance(predictions, pd.Series):
        pred_series = predictions
    else:
        import numpy as np

        pred_series = pd.Series(np.asarray(predictions).ravel(), index=live_df.index)

    if validate:
        pred_df = pred_series.rename("prediction").reset_index()
        if "index" in pred_df.columns:
            pred_df = pred_df.rename(columns={"index": "id"})
        issues = validate_submission(pred_df, live_df)
        if issues:
            for issue in issues:
                if issue.startswith("ERROR"):
                    logger.error(issue)
                else:
                    logger.warning(issue)
            errors = [i for i in issues if i.startswith("ERROR")]
            if errors:
                logger.error(
                    "Submission validation failed. Fix errors before submitting."
                )
                sys.exit(1)
        else:
            logger.info("Submission validation passed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission_df = pd.DataFrame(
        {
            "id": pred_series.index if "id" not in live_df.columns else live_df["id"],
            "prediction": pred_series.to_numpy(),
        }
    )
    submission_df.to_csv(output_path, index=False)

    logger.info(
        "Predictions saved to {} ({} rows, range=[{:.4f}, {:.4f}])",
        output_path,
        len(submission_df),
        float(submission_df["prediction"].min()),
        float(submission_df["prediction"].max()),
    )


if __name__ == "__main__":
    tyro.cli(main)
