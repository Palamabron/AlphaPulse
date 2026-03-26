from typing import Any, Self

import numpy as np
import pandas as pd

_SDV_AVAILABLE = False
try:
    from sdv.single_table import GaussianCopulaSynthesizer  # noqa: F401

    _SDV_AVAILABLE = True
except ImportError:
    pass


class SyntheticDataAugmenter:
    def __init__(
        self,
        top_fraction: float = 0.10,
        n_synthetic: int = 500,
        backend: str = "auto",
        seed: int = 42,
    ) -> None:
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
        if n_synthetic < 1:
            raise ValueError(f"n_synthetic must be >= 1, got {n_synthetic}")
        self.top_fraction = top_fraction
        self.n_synthetic = n_synthetic
        self.backend = backend
        self.seed = seed
        self._fitted = False
        self._elite_X: pd.DataFrame | None = None
        self._elite_y: pd.Series | None = None
        self._synthesizer: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        n_elite = max(1, int(len(y) * self.top_fraction))
        top_idx = y.nlargest(n_elite).index
        self._elite_X = X.loc[top_idx].copy().reset_index(drop=True)
        self._elite_y = y.loc[top_idx].copy().reset_index(drop=True)

        use_sdv = (self.backend == "sdv") or (self.backend == "auto" and _SDV_AVAILABLE)

        if use_sdv and _SDV_AVAILABLE:
            self._fit_sdv()
        elif self.backend == "sdv" and not _SDV_AVAILABLE:
            raise ImportError("sdv is required when backend='sdv'. pip install sdv")
        else:
            self._fit_kde()

        self._fitted = True
        return self

    def _fit_sdv(self) -> None:
        from sdv.metadata import Metadata
        from sdv.single_table import GaussianCopulaSynthesizer

        assert self._elite_X is not None and self._elite_y is not None
        elite_df = self._elite_X.copy()
        elite_df["__target__"] = self._elite_y.values

        metadata = Metadata.detect_from_dataframe(elite_df)
        synth = GaussianCopulaSynthesizer(
            metadata, enforce_min_max_values=True, enforce_rounding=False
        )
        synth.fit(elite_df)
        self._synthesizer = synth

    def _fit_kde(self) -> None:
        pass

    def generate(self) -> tuple[pd.DataFrame, pd.Series]:
        if not self._fitted:
            raise RuntimeError("Call fit() before generate().")
        assert self._elite_X is not None and self._elite_y is not None

        if self._synthesizer is not None and hasattr(self._synthesizer, "sample"):
            return self._generate_sdv()
        return self._generate_kde()

    def _generate_sdv(self) -> tuple[pd.DataFrame, pd.Series]:
        assert self._synthesizer is not None and self._elite_X is not None
        synth_df = self._synthesizer.sample(num_rows=self.n_synthetic)
        y_synth = synth_df.pop("__target__")
        return synth_df, y_synth

    def _generate_kde(self) -> tuple[pd.DataFrame, pd.Series]:
        from scipy.stats import gaussian_kde

        assert self._elite_X is not None and self._elite_y is not None
        rng = np.random.RandomState(self.seed)

        combined = np.column_stack([self._elite_X.values, self._elite_y.values])
        try:
            kde = gaussian_kde(combined.T, bw_method="silverman")
            samples = kde.resample(size=self.n_synthetic, seed=rng).T
        except np.linalg.LinAlgError:
            idx = rng.choice(len(combined), size=self.n_synthetic, replace=True)
            noise = rng.randn(self.n_synthetic, combined.shape[1]) * 0.01
            samples = combined[idx] + noise

        X_synth = pd.DataFrame(samples[:, :-1], columns=list(self._elite_X.columns))
        y_synth = pd.Series(samples[:, -1], name="target")
        return X_synth, y_synth

    @property
    def is_fitted(self) -> bool:
        return self._fitted
