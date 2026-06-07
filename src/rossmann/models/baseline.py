"""Baseline forecasters — the floor any real model must clear.

* ``NaiveLastWeek``  — predict the same store/day's sales from one week earlier.
* ``MedianBaseline`` — predict the per-(store, day-of-week) median from training.

The median baseline is what a competent business analyst would ship without ML;
beating it convincingly is the bar for the LightGBM model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rossmann import config
from rossmann.models.base import BaseForecaster

_STORE = config.COLS.store
_DOW = "DayOfWeek"
_TARGET = config.COLS.target


class MedianBaseline(BaseForecaster):
    """Per-(Store, DayOfWeek) median sales, with sensible fallbacks."""

    name = "median_baseline"

    def __init__(self) -> None:
        self._store_dow_median: pd.Series | None = None
        self._store_median: pd.Series | None = None
        self._global_median: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MedianBaseline":
        df = X[[_STORE, _DOW]].copy()
        df[_TARGET] = np.asarray(y)
        self._store_dow_median = df.groupby([_STORE, _DOW])[_TARGET].median()
        self._store_median = df.groupby(_STORE)[_TARGET].median()
        self._global_median = float(df[_TARGET].median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = list(zip(X[_STORE], X[_DOW]))
        preds = self._store_dow_median.reindex(keys).to_numpy()
        # Fallback 1: store median for unseen (store, dow) combos.
        store_fallback = self._store_median.reindex(X[_STORE]).to_numpy()
        preds = np.where(np.isnan(preds), store_fallback, preds)
        # Fallback 2: global median for entirely unseen stores.
        preds = np.where(np.isnan(preds), self._global_median, preds)
        return preds


class NaiveLastWeek(BaseForecaster):
    """Predict sales from 7 days earlier for the same store.

    Requires a ``Date`` column in ``X``. Falls back to the per-store mean from
    training when no 7-day-prior observation exists.
    """

    name = "naive_last_week"

    def __init__(self) -> None:
        self._history: pd.Series | None = None  # index: (Store, Date) -> Sales
        self._store_mean: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveLastWeek":
        df = X[[_STORE, config.COLS.date]].copy()
        df[_TARGET] = np.asarray(y)
        self._history = df.set_index([_STORE, config.COLS.date])[_TARGET]
        self._store_mean = df.groupby(_STORE)[_TARGET].mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prior_dates = X[config.COLS.date] - pd.Timedelta(days=7)
        keys = list(zip(X[_STORE], prior_dates))
        preds = self._history.reindex(keys).to_numpy()
        store_fallback = self._store_mean.reindex(X[_STORE]).to_numpy()
        preds = np.where(np.isnan(preds), store_fallback, preds)
        return np.nan_to_num(preds, nan=0.0)
