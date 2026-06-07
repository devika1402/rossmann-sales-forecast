"""Feature tests: shape/NaN sanity, Promo2 correctness, and — most importantly —
temporal correctness of lags across closed-day calendar gaps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rossmann.features import calendar, competition, lags, promotions


def test_calendar_adds_expected_columns(synthetic_raw):
    out = calendar.add_features(synthetic_raw)
    for col in ["Month", "WeekOfYear", "Quarter", "Year", "DayOfMonth", "DaysToChristmas"]:
        assert col in out.columns
    assert out["Month"].between(1, 12).all()
    assert out["DaysToChristmas"].between(0, 60).all()
    # Easter feature deliberately absent; generic holiday proximity present.
    assert "DaysToEaster" not in out.columns
    assert "DaysToPublicHoliday" in out.columns


def test_promo2_active_only_in_interval_months(synthetic_raw):
    df = calendar.add_features(synthetic_raw)
    df = promotions.add_features(df)
    # Store 2 participates (interval Jan,Apr,Jul,Oct); store 1 never does.
    store2 = df[df["Store"] == 2]
    jan = store2[store2["Date"].dt.month == 1]
    feb = store2[store2["Date"].dt.month == 2]
    assert jan["Promo2Active"].max() == 1  # active in January
    assert feb["Promo2Active"].max() == 0  # not active in February
    assert df[df["Store"] == 1]["Promo2Active"].max() == 0  # non-participant


def test_lag7_respects_calendar_across_closed_days(synthetic_raw):
    """The crux test: lag-7 must point to the true calendar day 7 days back,
    INCLUDING closed (zero-sales) days, not the previous *open* day."""
    df = lags.add_lag_features(synthetic_raw)
    s1 = df[df["Store"] == 1].set_index("Date").sort_index()
    # Pick a date far enough in for a full lag history.
    target_day = pd.Timestamp("2014-06-15")
    seven_back = target_day - pd.Timedelta(days=7)
    assert s1.loc[target_day, "SalesLag7"] == pytest.approx(s1.loc[seven_back, "Sales"])


def test_rolling_uses_shift_no_same_day_leak(synthetic_raw):
    df = lags.add_lag_features(synthetic_raw)
    s1 = df[df["Store"] == 1].set_index("Date").sort_index()
    day = pd.Timestamp("2014-06-15")
    prev7 = s1.loc[day - pd.Timedelta(days=7) : day - pd.Timedelta(days=1), "Sales"]
    # Rolling mean must equal the mean of the *previous* 7 days, excluding today.
    assert s1.loc[day, "SalesRollMean7"] == pytest.approx(prev7.mean(), rel=1e-5)


def test_store_aggregates_fallback_for_unseen_store(synthetic_raw):
    df = calendar.add_features(synthetic_raw)
    agg = lags.StoreAggregates().fit(df)
    # A frame with an unseen store id must get the global-mean fallback, no NaN.
    unseen = df.head(5).copy()
    unseen["Store"] = 999
    out = agg.transform(unseen)
    assert out["StoreSalesMean"].notna().all()
    assert "StoreDowAvgCustomers" in out.columns


def test_competition_cap_is_data_driven(synthetic_raw):
    out = competition.add_features(synthetic_raw)
    assert "CompetitionDistanceLog" in out.columns
    assert (out["CompetitionOpenMonths"] >= 0).all()
    assert "competition_open_months_cap" in out.attrs
