from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from alphapulse.hpo.builder import TREE_MODEL_NAMES, build_models
from alphapulse.models.era_ensemble_model import EraEnsembleModel
from alphapulse.models.packboost_model import PackboostModel
from alphapulse.models.sklearn_models import RidgeModel
from alphapulse.models.xgboost_model import XGBoostModel
from alphapulse.pipeline.pipeline import Pipeline
from alphapulse.preprocessors.scaling import StandardScalerPreprocessor

N_ERAS = 10
ROWS_PER_ERA = 20
N_FEATURES = 4


@pytest.fixture
def toy_data_with_era() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = N_ERAS * ROWS_PER_ERA
    X = pd.DataFrame(rng.standard_normal((n, N_FEATURES)), columns=list("ABCD"))
    X["era"] = np.repeat([f"era_{i:04d}" for i in range(N_ERAS)], ROWS_PER_ERA)
    y = pd.Series(rng.standard_normal(n))
    return X, y


@pytest.fixture
def xgb_factory() -> Callable[[], XGBoostModel]:
    def factory() -> XGBoostModel:
        return XGBoostModel(
            params={
                "max_depth": 3,
                "tree_method": "hist",
                "objective": "reg:squarederror",
            },
            name="XGB",
        )

    return factory


def test_era_ensemble_trains_n_sub_models(
    toy_data_with_era: tuple[pd.DataFrame, pd.Series],
    xgb_factory: Callable[[], XGBoostModel],
) -> None:
    X, y = toy_data_with_era
    model = EraEnsembleModel(xgb_factory, n_subs=4, name="EraXGB")
    model.train(X, y, n_rounds=5)
    assert len(model._sub_models) == 4


def test_era_ensemble_predict_shape(
    toy_data_with_era: tuple[pd.DataFrame, pd.Series],
    xgb_factory: Callable[[], XGBoostModel],
) -> None:
    X, y = toy_data_with_era
    model = EraEnsembleModel(xgb_factory, n_subs=3, name="EraXGB")
    model.train(X, y, n_rounds=5)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_era_ensemble_without_validation_uses_equal_average(
    toy_data_with_era: tuple[pd.DataFrame, pd.Series],
    xgb_factory: Callable[[], XGBoostModel],
) -> None:
    X, y = toy_data_with_era
    model = EraEnsembleModel(xgb_factory, n_subs=3, name="EraXGB")
    model.train(X, y, n_rounds=5)
    expected = np.mean(
        np.column_stack([submodel.predict(X) for submodel in model._sub_models]),
        axis=1,
    )

    assert model._meta_model is None
    np.testing.assert_allclose(model.predict(X), expected)


def test_era_ensemble_meta_model_uses_separate_validation_rows(
    toy_data_with_era: tuple[pd.DataFrame, pd.Series],
    xgb_factory: Callable[[], XGBoostModel],
) -> None:
    X, y = toy_data_with_era
    split = len(X) // 2
    model = EraEnsembleModel(xgb_factory, n_subs=3, name="EraXGB")
    model.train(
        X.iloc[:split],
        y.iloc[:split],
        X_val=X.iloc[split:],
        y_val=y.iloc[split:],
        n_rounds=5,
    )

    assert isinstance(model._meta_model, Ridge)


def test_era_ensemble_reserves_validation_for_meta_learner() -> None:
    calls: list[dict[str, object]] = []

    class RecordingModel(XGBoostModel):
        def train(  # type: ignore[override]
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None,
            **kwargs: object,
        ) -> dict[str, float]:
            calls.append(
                {
                    "X_val": X_val,
                    "y_val": y_val,
                    "early_stopping_rounds": kwargs.get("early_stopping_rounds"),
                }
            )
            self.is_trained = True
            return {}

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X), dtype=np.float64)

    X = pd.DataFrame(
        {"feature": np.arange(12, dtype=float), "era": np.repeat(["1", "2", "3"], 4)}
    )
    y = pd.Series(np.arange(12, dtype=float))
    model = EraEnsembleModel(lambda: RecordingModel(), n_subs=2)

    model.train(
        X.iloc[:8],
        y.iloc[:8],
        X_val=X.iloc[8:],
        y_val=y.iloc[8:],
        early_stopping_rounds=3,
    )

    assert calls
    assert all(call["X_val"] is None for call in calls)
    assert all(call["y_val"] is None for call in calls)
    assert all(call["early_stopping_rounds"] is None for call in calls)


def test_era_ensemble_fallback_no_era(xgb_factory: Callable[[], XGBoostModel]) -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("ABCD"))
    y = pd.Series(rng.standard_normal(100))
    model = EraEnsembleModel(xgb_factory, n_subs=5, name="EraXGB")
    model.train(X, y, n_rounds=5)
    assert len(model._sub_models) == 1
    assert model._meta_model is None
    assert model.predict(X).shape == (100,)


def test_era_ensemble_is_trained_flag(
    toy_data_with_era: tuple[pd.DataFrame, pd.Series],
    xgb_factory: Callable[[], XGBoostModel],
) -> None:
    X, y = toy_data_with_era
    model = EraEnsembleModel(xgb_factory, n_subs=3, name="EraXGB")
    assert not model.is_trained
    model.train(X, y, n_rounds=5)
    assert model.is_trained


def test_build_models_wraps_xgboost() -> None:
    config = [{"type": "XGBoost", "params": {}}]
    models = build_models(config)
    assert len(models) == 1
    assert isinstance(models[0], EraEnsembleModel)


def test_build_models_wraps_lightgbm() -> None:
    config = [{"type": "LightGBM", "params": {}}]
    models = build_models(config)
    assert isinstance(models[0], EraEnsembleModel)


def test_build_models_wraps_catboost() -> None:
    config = [{"type": "CatBoost", "params": {}}]
    models = build_models(config)
    assert isinstance(models[0], EraEnsembleModel)


def test_build_models_does_not_wrap_packboost() -> None:
    config = [{"type": "Packboost", "params": {}}]
    models = build_models(config)
    assert isinstance(models[0], PackboostModel)


def test_build_models_does_not_wrap_ridge() -> None:
    config = [{"type": "Ridge", "params": {}}]
    models = build_models(config)
    assert isinstance(models[0], RidgeModel)


def test_build_models_n_subs_forwarded() -> None:
    config = [{"type": "XGBoost", "params": {}, "n_subs": 7}]
    models = build_models(config)
    assert isinstance(models[0], EraEnsembleModel)
    assert models[0].n_subs == 7


def test_tree_model_names_set() -> None:
    assert "XGBoost" in TREE_MODEL_NAMES
    assert "LightGBM" in TREE_MODEL_NAMES
    assert "CatBoost" in TREE_MODEL_NAMES
    assert "RandomForest" in TREE_MODEL_NAMES
    assert "ExtraTrees" in TREE_MODEL_NAMES
    assert "Ridge" not in TREE_MODEL_NAMES
    assert "Packboost" not in TREE_MODEL_NAMES


def test_full_pipeline_with_era_ensemble(
    toy_data_with_era: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data_with_era

    def factory() -> XGBoostModel:
        return XGBoostModel(
            params={
                "max_depth": 3,
                "tree_method": "hist",
                "objective": "reg:squarederror",
            },
            name="XGB",
        )

    model = EraEnsembleModel(factory, n_subs=3, name="EraXGB")
    pipeline = Pipeline(
        preprocessors=[StandardScalerPreprocessor()],
        models=[model],
    )
    pipeline.fit(X, y, n_rounds=5)
    preds = pipeline.predict(X)
    assert preds.shape == (len(X),)
