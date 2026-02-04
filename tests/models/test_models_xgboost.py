import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphapulse.models.model_xgboost import ModelXgboost

ROOT = Path(__file__).parent.parent.parent
TRAIN_DATA_PATH = ROOT / "data" / "v5.2" / "train.parquet"
FEATURES_JSON_PATH = ROOT / "data" / "v5.2" / "features.json"
TEST_DATA_PATH = ROOT / "data" / "v5.2" / "live.parquet"


@pytest.fixture
def test_data() -> tuple[pd.DataFrame, list[str]]:
    """Load Numerai data"""
    with open(FEATURES_JSON_PATH, encoding="utf-8") as f:
        feature_metadata = json.load(f)
    feature_cols = feature_metadata["feature_sets"]["small"]
    target_cols = feature_metadata["targets"]
    train = pd.read_parquet(
        TRAIN_DATA_PATH, columns=["era"] + feature_cols + target_cols
    )
    return train, feature_cols


@pytest.fixture
def xgb_params() -> dict[str, Any]:
    return {
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1,
        "alpha": 0,
    }


def test_train_creates_model(
    test_data: tuple[pd.DataFrame, list[str]], xgb_params: dict[str, Any]
) -> None:
    """Checks if model was created"""
    train, feature_cols = test_data

    model = ModelXgboost()
    booster = model.train(
        train[feature_cols],
        train["target"],
        params=xgb_params,
        num_boost_round=10,
    )

    assert booster is not None
    assert model.model is booster


def test_finetune_updates_model(
    test_data: tuple[pd.DataFrame, list[str]], xgb_params: dict[str, Any]
) -> None:
    """Check if finetuning actually changes the model"""
    train, feature_cols = test_data

    model = ModelXgboost()

    booster_before = model.train(
        train[feature_cols],
        train["target"],
        params=xgb_params,
        num_boost_round=5,
    )

    booster_after = model.finetune(
        train[feature_cols],
        train["target"],
        params=xgb_params,
        num_boost_round=5,
    )

    assert booster_after is not None
    assert booster_after is not booster_before


def test_predict_output_shape_and_range(
    test_data: tuple[pd.DataFrame, list[str]], xgb_params: dict[str, Any]
) -> None:
    """Checks if the number of predictions is equal to the number of test samples
    and if each prediction is in [0,1]
    """
    train, feature_cols = test_data
    test = pd.read_parquet(TEST_DATA_PATH, columns=feature_cols)
    model = ModelXgboost()
    model.train(
        train[feature_cols],
        train["target"],
        params=xgb_params,
        num_boost_round=10,
    )

    preds = model.predict(test)

    assert preds.shape[0] == test.shape[0]
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
