"""Abstract forecaster interface shared by baseline, LightGBM and StatsForecast.

A common ``fit / predict / evaluate`` surface lets ``pipeline`` and the scripts
treat every model interchangeably.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd

from rossmann.evaluation import metrics


class BaseForecaster(ABC):
    """All models predict real (non-log) Sales from a feature frame."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """Fit on features ``X`` and real-sales target ``y``."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted real Sales for each row of ``X``."""

    def evaluate(self, X: pd.DataFrame, y_true) -> Dict[str, float]:
        """Predict on ``X`` and score against ``y_true`` with the metric bundle."""
        preds = self.predict(X)
        return metrics.evaluate(y_true, preds)
