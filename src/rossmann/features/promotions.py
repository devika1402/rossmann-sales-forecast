"""Promotion features — including the date-aware Promo2 decoding.

The single most common Rossmann mistake is treating the ``Promo2`` flag as if it
meant "promo running today". It does not: it only says the store *participates* in
the rolling Promo2 programme. A Promo2 cycle is only active in a given month if

    Promo2 == 1  AND  month in PromoInterval  AND  date >= Promo2 start date.

``add_features`` computes that correctly in ``Promo2Active``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rossmann import config

_DATE = config.COLS.date
_STORE = config.COLS.store


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add promo flags, the corrected Promo2Active flag, fatigue and interactions."""
    df = df.copy()

    df["Promo"] = df["Promo"].astype("int8")
    df["Promo2Active"] = _promo2_active(df).astype("int8")

    # Promotion fatigue: how many of the past 4 weeks (same weekday) ran a promo.
    df = _consecutive_promo_weeks(df)

    # Friday promos compound; encoding the interaction lets a single split capture it.
    df["PromoWeekdayInteraction"] = (df["Promo"] * df["DayOfWeek"]).astype("int16")
    return df


def _promo2_active(df: pd.DataFrame) -> pd.Series:
    """Boolean: is the store's Promo2 cycle active on this row's date?"""
    participates = df["Promo2"] == 1

    # Has the store's Promo2 programme started by this date?
    start = pd.to_datetime(
        df["Promo2SinceYear"].astype(int).astype(str) + "-1-1"
    ) + pd.to_timedelta((df["Promo2SinceWeek"].astype(int) - 1) * 7, unit="D")
    started = df[_DATE] >= start

    # Is the current calendar month one of the Promo2 restart months?
    # List comprehension over zip is far cheaper than a row-wise df.apply on ~1M rows.
    if "Promo2Months" in df.columns:
        months = df[_DATE].dt.month.to_numpy()
        month_sets = df["Promo2Months"].to_numpy()
        in_month = pd.Series(
            [m in s for m, s in zip(months, month_sets)], index=df.index
        )
    else:
        in_month = pd.Series(False, index=df.index)

    return participates & started & in_month


def _consecutive_promo_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling count of promo days over the trailing 28 days, per store.

    Uses ``shift(1)`` so the current day is excluded (no same-day leakage).
    """
    df = df.sort_values([_STORE, _DATE])
    df["ConsecutivePromoWeeks"] = (
        df.groupby(_STORE)["Promo"]
        .transform(lambda s: s.shift(1).rolling(window=28, min_periods=1).sum())
        .fillna(0)
        .astype("float32")
    )
    return df
