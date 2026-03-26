import numpy as np
import pandas as pd

from alphapulse.evaluation.metrics import era_sharpe
from alphapulse.validation import PurgedEraCV


def test_purged_cv_train_test_no_overlap() -> None:
    rng = np.random.RandomState(42)
    n_eras, rows = 15, 10
    n = n_eras * rows
    eras = pd.Series(np.repeat([f"era_{i:04d}" for i in range(n_eras)], rows))
    X = pd.DataFrame(rng.randn(n, 3).astype(np.float32), columns=["a", "b", "c"])
    X["era"] = eras.values
    y = pd.Series(rng.randn(n).astype(np.float32))

    cv = PurgedEraCV(n_splits=3, n_purge=1, n_embargo=1, min_train_eras=4)
    for train_idx, test_idx in cv.split(X, y, groups=eras):
        assert set(train_idx).isdisjoint(set(test_idx))


def test_era_sharpe_constant_predictions_is_negative_infinity() -> None:
    n = 200
    eras = pd.Series(np.repeat(["e1", "e2", "e3", "e4"], 50))
    y_true = pd.Series(np.random.randn(n))
    y_pred = np.ones(n)

    sharpe = era_sharpe(y_true, y_pred, eras)
    assert sharpe == float("-inf")
