import numpy as np
import pandas as pd
import pytest

from alphapulse.evaluation import Backtester
from alphapulse.evaluation.metrics import rank_normalize
from alphapulse.models import XGBoostModel
from alphapulse.pipeline import Pipeline
from alphapulse.preprocessors import StandardScalerPreprocessor


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(np.random.randn(n, 4).astype(np.float64), columns=list("ABCD"))
    y = pd.Series(X["A"] * 0.5 + X["B"] * 0.3 + np.random.randn(n) * 0.2)
    return X, y


@pytest.fixture
def fitted_pipeline(toy_data: tuple[pd.DataFrame, pd.Series]) -> Pipeline:
    X, y = toy_data
    pipeline = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model=XGBoostModel(
            params={
                "max_depth": 3,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "objective": "reg:squarederror",
            }
        ),
    )
    pipeline.fit(X, y, n_rounds=10)
    return pipeline


def test_pipeline_fit_predict_preserves_shape(
    fitted_pipeline: Pipeline, toy_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, _ = toy_data
    preds = fitted_pipeline.predict(X)
    assert preds.shape == (len(X),)
    assert isinstance(preds, np.ndarray)


def test_to_numerai_predict_signature(fitted_pipeline: Pipeline) -> None:
    predict_fn = fitted_pipeline.to_numerai_predict()
    live = pd.DataFrame(
        np.random.randn(10, 4).astype(np.float64),
        columns=fitted_pipeline.feature_columns,
    )
    bench = pd.DataFrame()
    out = predict_fn(live, bench)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 1)
    assert "prediction" in out.columns
    assert list(out.index) == list(live.index)
    # rank-normalized output should be in [0, 1]
    assert out["prediction"].min() >= 0.0
    assert out["prediction"].max() <= 1.0


def test_backtester_returns_expected_keys(
    toy_data: tuple[pd.DataFrame, pd.Series], fitted_pipeline: Pipeline
) -> None:
    X, y = toy_data
    era = pd.Series(np.repeat(["e1", "e2"], 100), index=X.index)
    backtester = Backtester(fitted_pipeline, feature_columns=list(X.columns))
    metrics = backtester.evaluate(X, y, era)
    expected = {
        "mean_per_era_correlation",
        "std_per_era_correlation",
        "sharpe",
        "correlation",
    }
    assert expected <= set(metrics.keys())


def test_pipeline_weighted_ensemble(toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = toy_data
    pipeline = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        models=[
            XGBoostModel(
                params={
                    "max_depth": 2,
                    "learning_rate": 0.1,
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                }
            ),
            XGBoostModel(
                params={
                    "max_depth": 3,
                    "learning_rate": 0.05,
                    "tree_method": "hist",
                    "objective": "reg:squarederror",
                }
            ),
        ],
        ensemble_method="weighted",
        ensemble_params={"weights": [0.5, 0.5]},
    )
    pipeline.fit(X, y, n_rounds=10)
    preds = pipeline.predict(X)
    assert preds.shape == (len(X),)
    assert isinstance(preds, np.ndarray)


def test_pipeline_model_and_models_mutually_exclusive(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    model = XGBoostModel(
        params={
            "max_depth": 2,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "objective": "reg:squarederror",
        }
    )
    with pytest.raises(ValueError, match="model or models"):
        Pipeline(
            preprocessors=[StandardScalerPreprocessor()],
            model=model,
            models=[model],
        )


def test_pipeline_single_model_normalizes_to_list(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    model = XGBoostModel(
        params={
            "max_depth": 2,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "objective": "reg:squarederror",
        }
    )
    pipeline = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model=model,
    )
    assert pipeline.models == [model]
    pipeline.fit(X, y, n_rounds=5)
    assert pipeline.predict(X).shape == (len(X),)


def test_neutralization_fallback_rank_normalizes(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    pipeline = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        model=XGBoostModel(
            params={
                "max_depth": 2,
                "learning_rate": 0.1,
                "tree_method": "hist",
                "objective": "reg:squarederror",
            }
        ),
        neutralize_proportion=0.5,
        neutralize_features=["missing_col"],
    )
    pipeline.fit(X, y, n_rounds=5)
    preds = pipeline.predict(X)
    ranked = rank_normalize(preds)
    np.testing.assert_allclose(preds, ranked, rtol=1e-5)
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0
