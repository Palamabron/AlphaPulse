import optuna

from .base import BasePreprocessor
from .compression import PCAPreprocessor
from .feature_selection import LGBMImportanceSelector, VarianceFeatureSelector
from .noise import GaussianNoiseInjector
from .scaling import RobustScalerPreprocessor, StandardScalerPreprocessor


class PreprocessorFactory:
    def __init__(self, n_features: int, *, prefix: str = "prep") -> None:
        self.n_features = n_features
        self.prefix = prefix

    def _p(self, name: str) -> str:
        return f"{self.prefix}_{name}"

    def suggest(self, trial: optuna.Trial) -> list[BasePreprocessor]:
        steps: list[BasePreprocessor] = []

        fs_strategy = trial.suggest_categorical(
            self._p("feature_selection"), ["none", "variance", "lgbm_importance"]
        )
        if fs_strategy == "variance":
            keep_frac = trial.suggest_categorical(
                self._p("var_keep_fraction"), [0.50, 0.75, 1.0]
            )
            if keep_frac < 1.0:
                steps.append(
                    VarianceFeatureSelector(keep_fraction=keep_frac, mode="quantile")
                )
        elif fs_strategy == "lgbm_importance":
            keep_frac = trial.suggest_categorical(
                self._p("lgbm_keep_fraction"), [0.50, 0.75, 1.0]
            )
            if keep_frac < 1.0:
                n_est = trial.suggest_int(
                    self._p("lgbm_n_estimators"), 50, 200, step=50
                )
                steps.append(
                    LGBMImportanceSelector(keep_fraction=keep_frac, n_estimators=n_est)
                )

        dim_red = trial.suggest_categorical(self._p("dim_reduction"), ["none", "pca"])
        if dim_red == "pca":
            max_comp = max(2, self.n_features // 2)
            min_comp = max(2, self.n_features // 10)
            n_components = trial.suggest_int(
                self._p("pca_n_components"), min_comp, max_comp
            )
            steps.append(PCAPreprocessor(n_components=n_components))

        scaler = trial.suggest_categorical(
            self._p("scaler"), ["standard", "robust", "none"]
        )
        if scaler == "standard":
            steps.append(StandardScalerPreprocessor())
        elif scaler == "robust":
            steps.append(RobustScalerPreprocessor())

        use_noise = trial.suggest_categorical(self._p("noise"), [True, False])
        if use_noise:
            sigma = trial.suggest_float(self._p("noise_sigma"), 0.001, 0.05, log=True)
            steps.append(GaussianNoiseInjector(sigma=sigma))

        return steps

    def suggest_fixed(
        self,
        feature_selection: str = "none",
        keep_fraction: float = 1.0,
        dim_reduction: str = "none",
        pca_n_components: int | None = None,
        scaler: str = "standard",
        noise: bool = False,
        noise_sigma: float = 0.01,
    ) -> list[BasePreprocessor]:
        steps: list[BasePreprocessor] = []

        if feature_selection == "variance" and keep_fraction < 1.0:
            steps.append(
                VarianceFeatureSelector(keep_fraction=keep_fraction, mode="quantile")
            )
        elif feature_selection == "lgbm_importance" and keep_fraction < 1.0:
            steps.append(LGBMImportanceSelector(keep_fraction=keep_fraction))

        if dim_reduction == "pca" and pca_n_components is not None:
            steps.append(PCAPreprocessor(n_components=pca_n_components))

        if scaler == "standard":
            steps.append(StandardScalerPreprocessor())
        elif scaler == "robust":
            steps.append(RobustScalerPreprocessor())

        if noise:
            steps.append(GaussianNoiseInjector(sigma=noise_sigma))

        return steps
