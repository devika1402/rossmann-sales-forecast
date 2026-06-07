"""Shared synthetic-data fixtures so the suite runs without the Kaggle CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    """A small, merged train-like frame: 3 stores x ~400 days, with closed days.

    Mirrors the post-loader/post-cleaner schema (store metadata already merged,
    Promo2Months decoded) so feature modules can consume it directly.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2014-01-01", "2015-02-15", freq="D")
    rows = []
    for store in (1, 2, 3):
        store_type = ["a", "b", "c"][store - 1]
        for d in dates:
            # Store closed on Sundays (dow 7) and a fixed maintenance day.
            dow = d.dayofweek + 1  # Rossmann: 1=Mon .. 7=Sun
            is_open = 0 if dow == 7 else 1
            promo = int(d.day <= 15)  # promo first half of month
            base = 5000 + store * 1000 + (1500 if dow == 5 else 0) + promo * 1200
            sales = 0 if is_open == 0 else int(base + rng.normal(0, 200))
            customers = 0 if is_open == 0 else int(sales / 9)
            rows.append(
                {
                    "Store": store,
                    "DayOfWeek": dow,
                    "Date": d,
                    "Sales": sales,
                    "Customers": customers,
                    "Open": is_open,
                    "Promo": promo,
                    "StateHoliday": "a" if (d.month == 1 and d.day == 1) else "0",
                    "SchoolHoliday": 0,
                    "StoreType": store_type,
                    "Assortment": "a",
                    "CompetitionDistance": 500.0 * store,
                    "CompetitionOpenSinceMonth": 1,
                    "CompetitionOpenSinceYear": 2013,
                    "Promo2": 1 if store == 2 else 0,
                    "Promo2SinceWeek": 1,
                    "Promo2SinceYear": 2014,
                    "PromoInterval": "Jan,Apr,Jul,Oct" if store == 2 else None,
                    "Promo2Months": frozenset({1, 4, 7, 10}) if store == 2 else frozenset(),
                }
            )
    return pd.DataFrame(rows)
