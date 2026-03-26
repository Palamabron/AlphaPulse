"""Export a trained pipeline as Numerai-ready `predict.pkl`.

This script is intended to be used after HPO:
1) run `scripts/hpo_pipeline.py` with your desired `train_subsample` (e.g. 1/8),
2) take `best_config.json` from the HPO output,
3) re-train that config on the same subsampled data,
4) export `predict.pkl` (and `pipeline.pkl`) for upload.
"""

import gc
import json
import sys
from pathlib import Path

import cloudpickle
import tyro
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphapulse.experiments.data import load_train_only_frame
from alphapulse.hpo.builder import build_pipeline_or_multi
from alphapulse.hpo.search_space import get_train_kwargs_from_flat, resolve_flat_config


def _needs_era_from_flat_config(flat: dict) -> bool:
    if bool(flat.get("use_packboost", False)):
        return True
    num_models = int(flat.get("num_models", 1))
    for i in range(1, min(num_models, 3) + 1):
        if flat.get(f"model_{i}_type") == "Packboost":
            return True
    return False


def _internal_val_split(X_train, y_train):
    """Match the internal early-stopping split used in HPO."""
    INTERNAL_VAL_THRESHOLD = 5000
    INTERNAL_VAL_FRACTION = 0.1
    if len(X_train) > INTERNAL_VAL_THRESHOLD:
        n_val_internal = int(len(X_train) * INTERNAL_VAL_FRACTION)
        return (
            X_train.iloc[:-n_val_internal],
            y_train.iloc[:-n_val_internal],
            X_train.tail(n_val_internal),
            y_train.tail(n_val_internal),
        )
    return X_train, y_train, None, None


def main(
    *,
    data_dir: Path = Path("data/v5.2"),
    best_config_path: Path = Path("artifacts/hpo/best_config.json"),
    train_subsample: float = 0.125,
    target_col: str = "target",
    seed: int = 42,
    output_dir: Path = Path("artifacts/competition_pickle_x8"),
) -> None:
    """Train best HPO config on subsampled data and export `predict.pkl`."""

    if not best_config_path.exists():
        raise FileNotFoundError(f"best_config_path not found: {best_config_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    with open(best_config_path, encoding="utf-8") as f:
        flat_config = json.load(f)
    if not isinstance(flat_config, dict):
        raise ValueError(f"Expected a JSON object in {best_config_path}")

    need_era = _needs_era_from_flat_config(flat_config)
    feature_set = flat_config.get("feature_set")

    logger.info(
        "Loading train data (subsample={}, feature_set={!r})...",
        train_subsample,
        feature_set,
    )
    X_train, y_train, feature_cols = load_train_only_frame(
        data_dir=data_dir,
        train_subsample=train_subsample,
        target_col=target_col,
        seed=seed,
        feature_columns=None,
        need_era=need_era,
        feature_set=feature_set,
    )
    gc.collect()

    mem_mb = X_train.memory_usage(deep=True).sum() / 1e6
    logger.info("Train shape: {} ({:.1f} MB)", X_train.shape, mem_mb)

    X_train_fit, y_train_fit, X_val_internal, y_val_internal = _internal_val_split(
        X_train, y_train
    )
    del X_train, y_train
    gc.collect()

    pipeline_config = resolve_flat_config(flat_config)
    pipeline = build_pipeline_or_multi(pipeline_config, feature_columns=feature_cols)
    train_kwargs = get_train_kwargs_from_flat(flat_config)

    logger.info("Fitting pipeline...")
    pipeline.fit(
        X_train_fit,
        y_train_fit,
        X_val=X_val_internal,
        y_val=y_val_internal,
        **train_kwargs,
    )

    del X_train_fit, y_train_fit, X_val_internal, y_val_internal
    gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_pipeline_config.json").write_text(
        json.dumps(pipeline_config, indent=2),
        encoding="utf-8",
    )

    predict_fn = pipeline.to_numerai_predict()
    with open(output_dir / "predict.pkl", "wb") as f:
        cloudpickle.dump(predict_fn, f)

    pipeline.save_pipeline(output_dir / "pipeline.pkl")

    logger.info("Exported Numerai predict to: {}", output_dir / "predict.pkl")
    logger.info("Saved trained pipeline to:   {}", output_dir / "pipeline.pkl")


if __name__ == "__main__":
    tyro.cli(main)
