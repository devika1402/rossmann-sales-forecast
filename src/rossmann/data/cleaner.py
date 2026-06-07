"""Cleaning: impute store-metadata nulls and decode the Promo2 interval.

IMPORTANT — what this module does NOT do: it does not drop closed / zero-sales
rows. That filtering happens in ``pipeline`` *after* lag features are built, so
that lags reference true calendar days rather than the previous *open* day. See
``pipeline.filter_trainable``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rossmann import config

logger = logging.getLogger(__name__)

# Map the textual PromoInterval ("Jan,Apr,Jul,Oct") to the set of month numbers
# in which a Promo2 cycle restarts.
_MONTH_ABBR_TO_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _decode_promo_interval(interval: object) -> frozenset:
    if not isinstance(interval, str) or not interval:
        return frozenset()
    return frozenset(_MONTH_ABBR_TO_NUM[m] for m in interval.split(",") if m in _MONTH_ABBR_TO_NUM)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute nulls and add the decoded ``Promo2Months`` set column.

    Returns a copy; the input frame is left untouched.
    """
    df = df.copy()

    # Competition distance: null almost always means "no competitor in catchment",
    # so a large sentinel is more honest than the median (which understates it).
    n_missing_dist = int(df["CompetitionDistance"].isna().sum())
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(config.NO_COMPETITOR_DISTANCE)

    # Competition open date: null => treat as "always existed" (1900-01).
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(1900).astype("int32")
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(1).astype("int32")

    # Promo2 start: null => store never ran Promo2; sentinel year keeps it inactive.
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(2100).astype("int32")
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(1).astype("int32")

    # Decode interval string -> set of restart months (used by promotions feature).
    df["Promo2Months"] = df["PromoInterval"].map(_decode_promo_interval)

    # Categorical store metadata: fill the rare missing with a dedicated token.
    for col in ("StoreType", "Assortment"):
        df[col] = df[col].fillna("missing").astype(str)

    logger.info("Cleaned frame: imputed %d missing CompetitionDistance values", n_missing_dist)
    assert df["CompetitionDistance"].notna().all(), "CompetitionDistance still has nulls"
    return df


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p(Sales) target column where Sales exists (train only)."""
    df = df.copy()
    if config.COLS.target in df.columns:
        df[config.COLS.log_target] = np.log1p(df[config.COLS.target].clip(lower=0))
    return df
