from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import xgboost as xgb


class ModelAbstract(ABC):
    """Abstract class for all models"""

    @abstractmethod
    def train(self, *_args: Any, **_kwargs: Any) -> xgb.Booster:
        """Initial training the model"""
        raise NotImplementedError("Train method needs to be overriden")

    @abstractmethod
    def finetune(self, *_args: Any, **_kwargs: Any) -> xgb.Booster:
        """Finetune the trained model"""
        raise NotImplementedError("Finetune method needs to be overriden")

    @abstractmethod
    def predict(self, *_args: Any, **_kwargs: Any) -> pd.Series:
        """Predict the result of the trained model"""
        raise NotImplementedError("Predict method needs to be overriden")
