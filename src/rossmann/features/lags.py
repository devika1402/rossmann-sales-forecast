"""Lag / rolling features and the fit/transform ``StoreAggregates``.

Two distinct kinds of "history" features live here:

1. Row-level lags (``add_lag_features``) — same-store sales N days ago, and
   trailing rolling stats. These are computed on the FULL calendar (before any
   closed-day filtering) so that ``sales_lag_7`` references the true 7-days-prior
   calendar day, not the previous *open* day. All rolling windows use ``shift(1)``
   to exclude the current day.

2. Store-level aggregates (``StoreAggregates``) — per-store and per-(store, dow)
   summaries. These follow the sklearn fit/transform contract and MUST be fitted
   on the training split only, then applied to validation/test. Fitting on the
   full frame is a classic leakage source. Includes ``store_dow_avg_customers`` —
   a proxy that carries the (train-only) ``Customers`` footfall signal into test
   time, where the raw Customers column does not exist.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from rossmann import config

_STORE = config.COLS.store
_DATE = config.COLS.date
_TARGET = config.COLS.target

LAG_DAYS: List[int] = [7, 14, 28, 365]
ROLLING_WINDOWS: List[int] = [7, 28]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add same-store sales lags and trailing rolling mean/std.

    Must be called BEFORE closed/zero-sales rows are filtered out.
    NaNs (early history with no lag available) are left as-is — LightGBM handles
    them natively, and the first ``max(LAG_DAYS)`` days are dropped in training.
    """
    df = df.sort_values([_STORE, _DATE]).reset_index(drop=True)
    grp = df.groupby(_STORE)[_TARGET]

    for lag in LAG_DAYS:
        df[f"SalesLag{lag}"] = grp.shift(lag).astype("float32")

    for window in ROLLING_WINDOWS:
        # shift(1) first => the window ends yesterday, never includes today.
        shifted = grp.shift(1)
        df[f"SalesRollMean{window}"] = (
            shifted.rolling(window, min_periods=1).mean().astype("float32")
        )
    df["SalesRollStd7"] = grp.shift(1).rolling(7, min_periods=2).std().astype("float32")
    return df


class StoreAggregates:
    """Per-store summary features, fitted on train and applied to any split.

    Usage::

        agg = StoreAggregates().fit(train_df)
        train_df = agg.transform(train_df)
        val_df   = agg.transform(val_df)
    """

    def __init__(self) -> None:
        self._store_mean: pd.Series | None = None
        self._store_median: pd.Series | None = None
        self._store_dow_mean: pd.DataFrame | None = None
        self._store_dow_customers: pd.DataFrame | None = None
        self._store_promo_ratio: pd.Series | None = None
        self._global_mean: float = 0.0

    def fit(self, df: pd.DataFrame) -> "StoreAggregates":
        """Learn aggregates from a training frame (must contain Sales/Customers)."""
        self._global_mean = float(df[_TARGET].mean())

        self._store_mean = df.groupby(_STORE)[_TARGET].mean().rename("StoreSalesMean")
        self._store_median = df.groupby(_STORE)[_TARGET].median().rename("StoreSalesMedian")

        self._store_dow_mean = (
            df.groupby([_STORE, "DayOfWeek"])[_TARGET]
            .mean()
            .rename("StoreDowMean")
            .reset_index()
        )

        # Customers proxy — only meaningful at fit time (train has Customers).
        if "Customers" in df.columns:
            self._store_dow_customers = (
                df.groupby([_STORE, "DayOfWeek"])["Customers"]
                .mean()
                .rename("StoreDowAvgCustomers")
                .reset_index()
            )

        self._store_promo_ratio = self._compute_promo_ratio(df)
        return self

    @staticmethod
    def _compute_promo_ratio(df: pd.DataFrame) -> pd.Series:
        """mean(Sales | promo) / mean(Sales | no promo) per store; 1.0 if undefined."""
        promo_mean = df[df["Promo"] == 1].groupby(_STORE)[_TARGET].mean()
        base_mean = df[df["Promo"] == 0].groupby(_STORE)[_TARGET].mean()
        ratio = (promo_mean / base_mean).replace([np.inf, -np.inf], np.nan)
        return ratio.rename("StorePromoSalesRatio")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join learned aggregates onto ``df`` with global-mean fallbacks."""
        if self._store_mean is None:
            raise RuntimeError("StoreAggregates.transform called before fit().")
        df = df.copy()

        df["StoreSalesMean"] = (
            df[_STORE].map(self._store_mean).fillna(self._global_mean).astype("float32")
        )
        df["StoreSalesMedian"] = (
            df[_STORE].map(self._store_median).fillna(self._global_mean).astype("float32")
        )
        df = df.merge(self._store_dow_mean, on=[_STORE, "DayOfWeek"], how="left")
        df["StoreDowMean"] = df["StoreDowMean"].fillna(self._global_mean).astype("float32")

        if self._store_dow_customers is not None:
            df = df.merge(self._store_dow_customers, on=[_STORE, "DayOfWeek"], how="left")
            global_cust = float(self._store_dow_customers["StoreDowAvgCustomers"].mean())
            df["StoreDowAvgCustomers"] = (
                df["StoreDowAvgCustomers"].fillna(global_cust).astype("float32")
            )

        df["StorePromoSalesRatio"] = (
            df[_STORE].map(self._store_promo_ratio).fillna(1.0).astype("float32")
        )
        return df
