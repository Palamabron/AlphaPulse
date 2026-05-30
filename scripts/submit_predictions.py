"""Upload Numerai predictions via NumerAPI.

Requires NUMERAI_PUBLIC_API_KEY and NUMERAI_PRIVATE_API_KEY in the environment
or a .env file.

Usage:
    uv run python scripts/submit_predictions.py \\
        --predictions-path artifacts/live/predictions.csv \\
        --model-name my_model_name
"""

import os
import sys
from pathlib import Path

import pandas as pd
import tyro
from loguru import logger

from alphapulse.evaluation.submission import validate_submission


def _load_api_keys() -> tuple[str, str]:
    pub = os.environ.get("NUMERAI_PUBLIC_API_KEY", "")
    priv = os.environ.get("NUMERAI_PRIVATE_API_KEY", "")
    if not pub or not priv:
        try:
            from dotenv import load_dotenv

            load_dotenv()
            pub = os.environ.get("NUMERAI_PUBLIC_API_KEY", "")
            priv = os.environ.get("NUMERAI_PRIVATE_API_KEY", "")
        except ImportError:
            pass
    if not pub or not priv:
        logger.error(
            "NUMERAI_PUBLIC_API_KEY and NUMERAI_PRIVATE_API_KEY must be set "
            "in environment or .env file."
        )
        sys.exit(1)
    return pub, priv


def main(
    predictions_path: Path,
    model_name: str,
    tournament: str = "numerai",
    validate: bool = True,
) -> None:
    """Upload predictions to Numerai.

    Args:
        predictions_path: Path to a CSV with ``id`` and ``prediction`` columns.
        model_name: Your Numerai model name (as shown in the dashboard).
        tournament: Tournament identifier. Default: "numerai".
        validate: Run format validation before uploading.
    """
    try:
        import numerapi  # type: ignore[import]
    except ImportError:
        logger.error("numerapi is required. Install with: pip install numerapi")
        sys.exit(1)

    if not predictions_path.exists():
        logger.error("Predictions file not found: {}", predictions_path)
        sys.exit(1)

    predictions_df = pd.read_csv(predictions_path)
    logger.info(
        "Loaded {} predictions from {}",
        len(predictions_df),
        predictions_path,
    )

    if validate:
        issues = validate_submission(predictions_df)
        if issues:
            for issue in issues:
                if issue.startswith("ERROR"):
                    logger.error(issue)
                else:
                    logger.warning(issue)
            errors = [i for i in issues if i.startswith("ERROR")]
            if errors:
                logger.error("Validation failed. Fix errors before submitting.")
                sys.exit(1)
        else:
            logger.info("Submission validation passed.")

    pub_key, priv_key = _load_api_keys()
    napi = numerapi.NumerAPI(public_id=pub_key, secret_key=priv_key)

    # Look up model ID from model name
    models = napi.get_models()
    model_id = models.get(model_name)
    if model_id is None:
        available = list(models.keys())
        logger.error(
            "Model '{}' not found. Available models: {}", model_name, available
        )
        sys.exit(1)

    current_round = napi.get_current_round()
    logger.info(
        "Submitting {} predictions for model '{}' (id={}) to round {}...",
        len(predictions_df),
        model_name,
        model_id,
        current_round,
    )

    submission_id = napi.upload_predictions(
        str(predictions_path), model_id=model_id, tournament=tournament
    )
    logger.info(
        "Submission successful! submission_id={}, round={}",
        submission_id,
        current_round,
    )


if __name__ == "__main__":
    tyro.cli(main)
