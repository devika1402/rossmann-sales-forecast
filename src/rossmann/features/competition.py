"""Competition features.

Retail intuition:
* Distance effect is logarithmic — a rival 200m away hurts far more than one 2km
  away, but 8km vs 10km barely differs. Hence ``log1p`` of the distance.
* A newly opened competitor causes a step-down in sales that recovers slowly, so
  "months since the competitor opened" carries signal. The cap is data-driven
  (max observed in training + 12 months) rather than an arbitrary constant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rossmann import config

_DATE = config.COLS.date


def add_features(df: pd.DataFrame, open_months_cap: int | None = None) -> pd.DataFrame:
    """Add log-distance, competitor-age, and has-competitor flag.

    ``open_months_cap`` should be derived from the *training* split and passed in
    so train and validation share the same cap (no leakage, no surprises).
    """
    df = df.copy()

    df["CompetitionDistanceLog"] = np.log1p(df["CompetitionDistance"]).astype("float32")
    df["HasCompetitor"] = (
        df["CompetitionDistance"] < config.HAS_COMPETITOR_THRESHOLD
    ).astype("int8")

    months = _competition_open_months(df)
    if open_months_cap is None:
        # +12 month buffer beyond the largest observed age (documented choice).
        open_months_cap = int(months.max()) + 12
    df["CompetitionOpenMonths"] = months.clip(lower=0, upper=open_months_cap).astype("int16")
    df.attrs["competition_open_months_cap"] = open_months_cap
    return df


def _competition_open_months(df: pd.DataFrame) -> pd.Series:
    """Whole months between competitor opening and the row date (0 if not yet open)."""
    open_year = df["CompetitionOpenSinceYear"]
    open_month = df["CompetitionOpenSinceMonth"]
    months = (df[_DATE].dt.year - open_year) * 12 + (df[_DATE].dt.month - open_month)
    return months
