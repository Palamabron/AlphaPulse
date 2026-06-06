"""Minimal train + backtest + wandb + save Numerai-ready pickle."""

from pathlib import Path

import pandas as pd
import tyro

from alphapulse.evaluation import Backtester
from alphapulse.experiments.data import load_feature_names
from alphapulse.logging_ import init_wandb, log_backtest_results, log_metrics
from alphapulse.models.xgboost_model import XGBoostModel
from alphapulse.pipeline import Pipeline
from alphapulse.preprocessors.scaling import StandardScalerPreprocessor
from alphapulse.utils import set_global_seed


def main(
    data_dir: Path = Path("data/v5.2"),
    train_subsample: float = 0.05,
    target_col: str = "target",
    seed: int = 42,
    output_dir: Path = Path("artifacts"),
    wandb_project: str = "alphapulse-test",
    wandb_run_name: str | None = None,
) -> None:
    """Run test pipeline: load data, train, backtest, log to wandb, save pickle."""
    set_global_seed(seed)
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "validation.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Data not found. Expected {train_path} and {val_path}. "
            "Run scripts/download_dataset.py first."
        )

    feature_names = load_feature_names(data_dir)
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    if feature_names:
        feature_cols = [c for c in feature_names if c in train_df.columns]
    else:
        meta = {"id", "era", "target"}
        meta.update(c for c in train_df.columns if c.startswith("target_"))
        feature_cols = [
            c
            for c in train_df.columns
            if c not in meta and train_df[c].dtype in ("float64", "float32")
        ]
    if not feature_cols:
        raise ValueError("No feature columns found.")

    train_subsampled = train_df.sample(frac=train_subsample, random_state=42)
    X_train = train_subsampled[feature_cols]
    y_train = train_subsampled[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    era_val = val_df["era"]

    if len(X_train) > 5000:
        n_val_internal = int(len(X_train) * 0.1)
        X_val_internal = X_train.tail(n_val_internal)
        y_val_internal = y_train.tail(n_val_internal)
        X_train_fit = X_train.iloc[:-n_val_internal]
        y_train_fit = y_train.iloc[:-n_val_internal]
    else:
        X_val_internal = None
        y_val_internal = None
        X_train_fit = X_train
        y_train_fit = y_train

    xgb_params = {
        "max_depth": 5,
        "learning_rate": 0.01,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }
    pipeline = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model=XGBoostModel(params=xgb_params),
    )
    train_metrics = pipeline.fit(
        X_train_fit,
        y_train_fit,
        X_val=X_val_internal,
        y_val=y_val_internal,
        n_rounds=500,
        early_stopping_rounds=50,
    )

    backtester = Backtester(pipeline, feature_columns=feature_cols)
    backtest_metrics = backtester.evaluate(X_val, y_val, era_val)

    config = {
        "train_subsample": train_subsample,
        "n_features": len(feature_cols),
        "xgb_params": xgb_params,
    }
    init_wandb(project=wandb_project, config=config, name=wandb_run_name)
    log_metrics(train_metrics)
    log_backtest_results(backtest_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    predict_fn = pipeline.to_numerai_predict()
    import cloudpickle

    with open(output_dir / "predict.pkl", "wb") as f:
        cloudpickle.dump(predict_fn, f)
    pipeline.save_pipeline(output_dir / "pipeline.pkl")
    pkl = output_dir / "predict.pkl"
    pipe = output_dir / "pipeline.pkl"
    print(f"Saved Numerai predict to {pkl}, pipeline to {pipe}")
    print("Backtest metrics:", backtest_metrics)


if __name__ == "__main__":
    tyro.cli(main)
