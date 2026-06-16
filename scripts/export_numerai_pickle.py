"""Export a trained pipeline as Numerai-ready `predict.pkl`.

This script is intended to be used after HPO:
1) run `scripts/hpo_pipeline.py` with your desired `train_subsample` (e.g. 1/8),
2) take `best_config.json` from the HPO output,
3) re-train that config on the same subsampled data,
4) export `predict.pkl` (and `pipeline.pkl`) for upload.
"""

import datetime
import gc
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cloudpickle
import tyro
from loguru import logger

from alphapulse.evaluation.export_validation import smoke_test_predict_fn
from alphapulse.hpo.export import build_hpo_pipeline_from_flat
from alphapulse.utils import set_global_seed


def _artifact_stem(flat_config: dict[str, Any], target_col: str) -> str:
    """Build canonical artifact stem: TIMESTAMP_ARCH_TARGET_HASH."""
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
    arch = str(flat_config.get("model_1_type", "unknown"))
    config_hash = hashlib.sha1(  # noqa: S324
        json.dumps(flat_config, sort_keys=True).encode()
    ).hexdigest()[:8]
    return f"{ts}_{arch}_{target_col}_{config_hash}"


def _provenance(
    flat_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    target_col: str,
) -> dict[str, Any]:
    """Build a hermetically sealed provenance record."""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = "unavailable"

    try:
        dependencies = subprocess.check_output(["uv", "export", "--no-dev"], text=True)
    except Exception:
        dependencies = "unavailable"

    return {
        "git_commit": git_commit,
        "target_col": target_col,
        "flat_config": flat_config,
        "resolved_config": pipeline_config,
        "dependencies": dependencies,
    }


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

    logger.info(
        "Building pipeline from HPO config "
        "(subsample={}, routing={}, target_mode={})...",
        train_subsample,
        flat_config.get("use_feature_routing", False),
        flat_config.get("target_mode", "single"),
    )
    fit_result = build_hpo_pipeline_from_flat(
        flat_config,
        data_dir,
        train_subsample=train_subsample,
        seed=seed,
        target_col_fallback=target_col,
        allow_target_resample=False,
    )
    pipeline = fit_result.pipeline
    feature_cols = fit_result.feature_columns
    pipeline_config = fit_result.pipeline_cfg
    flat_config = fit_result.flat
    primary_target = fit_result.primary_target

    gc.collect()
    logger.info(
        "Pipeline fitted: {} features, primary_target={!r}",
        len(feature_cols),
        primary_target,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = _artifact_stem(flat_config, primary_target)

    (output_dir / "resolved_pipeline_config.json").write_text(
        json.dumps(pipeline_config, indent=2),
        encoding="utf-8",
    )

    prov = _provenance(flat_config, pipeline_config, primary_target)
    prov_path = output_dir / f"{stem}_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    logger.info("Provenance bundle saved to: {}", prov_path)

    predict_fn = pipeline.to_numerai_predict()
    pkl_path = output_dir / f"{stem}_predict.pkl"
    with open(pkl_path, "wb") as f:
        cloudpickle.dump(predict_fn, f)

    smoke_test_predict_fn(pkl_path, feature_cols)
    logger.info("Smoke test passed for {}", pkl_path)

    pipeline_pkl_path = output_dir / f"{stem}_pipeline.pkl"
    pipeline.save_pipeline(pipeline_pkl_path)

    latest_predict = output_dir / "latest_predict.pkl"
    if latest_predict.is_symlink() or latest_predict.exists():
        latest_predict.unlink()
    latest_predict.symlink_to(pkl_path.name)

    logger.info("Exported Numerai predict to: {}", pkl_path)
    logger.info("Saved trained pipeline to:   {}", pipeline_pkl_path)
    logger.info("Symlink updated:             {}", latest_predict)


if __name__ == "__main__":
    tyro.cli(main)
