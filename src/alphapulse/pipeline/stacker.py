from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.metrics import era_sharpe
from ..models.factory import ModelFactory
from ..preprocessors.factory import PreprocessorFactory
from ..validation.purged_cv import PurgedEraCV


class Stacker:
    def __init__(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        eras: pd.Series,
        cv: PurgedEraCV,
        n_rounds: int = 200,
        early_stopping_rounds: int = 30,
    ) -> None:
        self.X = X
        self.y = y
        self.eras = eras
        self.cv = cv
        self.n_rounds = n_rounds
        self.early_stopping_rounds = early_stopping_rounds

    def collect_oof(
        self,
        trial_params_list: list[dict[str, Any]],
        include_dl: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, pd.Series]:
        n_trials = len(trial_params_list)
        n_rows = len(self.X)
        oof_preds = np.full((n_rows, n_trials), np.nan, dtype=np.float64)
        covered = np.zeros(n_rows, dtype=bool)

        prep_factory = PreprocessorFactory(n_features=self.X.shape[1])
        model_factory = ModelFactory(include_dl=include_dl)

        for k, params in enumerate(trial_params_list):
            for _fold_i, (train_idx, test_idx) in enumerate(
                self.cv.split(self.X, self.y, groups=self.eras)
            ):
                X_train = self.X.iloc[train_idx]
                y_train = self.y.iloc[train_idx]
                X_test = self.X.iloc[test_idx]

                preprocessors = prep_factory.suggest_fixed(
                    feature_selection=params.get("prep_feature_selection", "none"),
                    keep_fraction=params.get(
                        "prep_var_keep_fraction",
                        params.get("prep_lgbm_keep_fraction", 1.0),
                    ),
                    dim_reduction=params.get("prep_dim_reduction", "none"),
                    pca_n_components=params.get("prep_pca_n_components"),
                    scaler=params.get("prep_scaler", "standard"),
                    noise=params.get("prep_noise", False),
                    noise_sigma=params.get("prep_noise_sigma", 0.01),
                )

                X_tr_pp = X_train.copy()
                for pp in preprocessors:
                    pp.fit(X_tr_pp, y_train)
                    X_tr_pp = pp.transform(X_tr_pp)

                X_te_pp = X_test.copy()
                for pp in preprocessors:
                    X_te_pp = pp.transform(X_te_pp)

                model_type = params.get("model_type", "LightGBM")
                model = model_factory.suggest_fixed(model_type)
                model.train(
                    X_tr_pp,
                    y_train,
                    n_rounds=self.n_rounds,
                    early_stopping_rounds=self.early_stopping_rounds,
                )

                oof_preds[test_idx, k] = model.predict(X_te_pp)
                covered[test_idx] = True

        mask = covered & np.all(np.isfinite(oof_preds), axis=1)
        return (
            oof_preds[mask],
            np.asarray(self.y.values[mask], dtype=np.float64),
            self.eras.iloc[np.where(mask)[0]].reset_index(drop=True),
        )

    def score_individual(
        self,
        oof_matrix: np.ndarray,
        y_oof: np.ndarray,
        eras_oof: pd.Series,
    ) -> list[float]:
        y_series = pd.Series(y_oof)
        return [
            era_sharpe(y_series, oof_matrix[:, k], eras_oof)
            for k in range(oof_matrix.shape[1])
        ]
