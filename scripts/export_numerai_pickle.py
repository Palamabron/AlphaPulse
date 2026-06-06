"""Export a trained pipeline as Numerai-ready `predict.pkl`.

This script is intended to be used after HPO:
1) run `scripts/hpo_pipeline.py` with your desired `train_subsample` (e.g. 1/8),
2) take `best_config.json` from the HPO output,
3) re-train that config on the same subsampled data,
4) export `predict.pkl` (and `pipeline.pkl`) for upload.
"""

import gc
import json
from pathlib import Path

import cloudpickle
import tyro
from loguru import logger

from alphapulse.evaluation.export_validation import smoke_test_predict_fn
from alphapulse.experiments.data import load_train_only_frame
from alphapulse.experiments.split import internal_val_split
from alphapulse.hpo.builder import TREE_MODEL_NAMES, build_pipeline_or_multi
from alphapulse.hpo.search_space import get_train_kwargs_from_flat, resolve_flat_config
from alphapulse.utils import set_global_seed


def _needs_era_from_flat_config(flat: dict) -> bool:
    if bool(flat.get("use_packboost", False)):
        return True
    num_models = int(flat.get("num_models", 1))
    for i in range(1, min(num_models, 3) + 1):
        model_type = flat.get(f"model_{i}_type", "")
        if model_type == "Packboost" or model_type in TREE_MODEL_NAMES:
            return True
    return False


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
    set_global_seed(seed)

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

    era_train = X_train["era"] if "era" in X_train.columns else None
    stacking_needs_val = (
        int(flat_config.get("num_models", 1)) > 1
        and flat_config.get("ensemble_method") == "stacking"
    )
    X_train_fit, y_train_fit, X_val_internal, y_val_internal = internal_val_split(
        X_train,
        y_train,
        era_train=era_train,
        force_internal=stacking_needs_val,
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
    pkl_path = output_dir / "predict.pkl"
    with open(pkl_path, "wb") as f:
        cloudpickle.dump(predict_fn, f)

    smoke_test_predict_fn(pkl_path, feature_cols)
    logger.info("Smoke test passed for {}", pkl_path)

    pipeline.save_pipeline(output_dir / "pipeline.pkl")

    logger.info("Exported Numerai predict to: {}", pkl_path)
    logger.info("Saved trained pipeline to:   {}", output_dir / "pipeline.pkl")


if __name__ == "__main__":
    tyro.cli(main)
