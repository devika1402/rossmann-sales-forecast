"""Calendar & holiday features — pure date arithmetic, no row ordering needed.

Retail intuition encoded here:
* Day-of-week and month capture the dominant weekly and seasonal sales rhythm.
* Day-of-month / month boundaries proxy the pay-day spending spike.
* Distance-to-Christmas is a non-linear ramp that saturates before the day itself.
* ``days_to_easter`` is deliberately NOT used (only 3 events in 2013-2015 — noise);
  it is folded into a generic ``days_to_public_holiday`` instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rossmann import config

_DATE = config.COLS.date


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar + holiday-proximity features. Returns a copy."""
    df = df.copy()
    d = df[_DATE].dt

    df["Month"] = d.month.astype("int16")
    df["WeekOfYear"] = d.isocalendar().week.astype("int16").to_numpy()
    df["Quarter"] = d.quarter.astype("int16")
    df["Year"] = d.year.astype("int16")
    df["DayOfMonth"] = d.day.astype("int16")
    df["IsMonthStart"] = d.is_month_start.astype("int8")
    df["IsMonthEnd"] = d.is_month_end.astype("int8")

    # Non-linear ramp into Christmas: days remaining, capped at 60.
    christmas = pd.to_datetime(df["Year"].astype(str) + "-12-25")
    df["DaysToChristmas"] = (christmas - df[_DATE]).dt.days.clip(lower=0, upper=60).astype("int16")

    # Generic holiday-proximity features (replaces brittle per-holiday Easter flag).
    df = _add_holiday_proximity(df)
    return df


def _add_holiday_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Days since/until the nearest public holiday (StateHoliday != '0').

    Computed per store on the sorted calendar so that gaps are handled correctly.
    """
    df = df.sort_values([config.COLS.store, _DATE]).reset_index(drop=True)
    is_holiday = (df["StateHoliday"] != "0").to_numpy()

    days_since = np.full(len(df), 999, dtype="int32")
    days_until = np.full(len(df), 999, dtype="int32")

    for _, idx in df.groupby(config.COLS.store).groups.items():
        idx = np.asarray(idx)
        dates = df.loc[idx, _DATE].to_numpy()
        hol = is_holiday[idx]
        holiday_dates = dates[hol]
        if holiday_dates.size == 0:
            continue
        # Vectorised nearest-previous / nearest-next holiday via searchsorted.
        pos = np.searchsorted(holiday_dates, dates, side="right")
        for j, (dt, p) in enumerate(zip(dates, pos)):
            if p > 0:
                days_since[idx[j]] = (dt - holiday_dates[p - 1]) / np.timedelta64(1, "D")
            if p < holiday_dates.size:
                days_until[idx[j]] = (holiday_dates[p] - dt) / np.timedelta64(1, "D")

    df["DaysSinceHoliday"] = np.clip(days_since, 0, 999)
    df["DaysToPublicHoliday"] = np.clip(days_until, 0, 999)
    return df


def encode_state_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """Map StateHoliday {0,a,b,c} -> {0,1,2,3} as an int feature for LightGBM."""
    df = df.copy()
    mapping = {"0": 0, "a": 1, "b": 2, "c": 3}
    df["StateHolidayEnc"] = df["StateHoliday"].map(mapping).fillna(0).astype("int8")
    return df
