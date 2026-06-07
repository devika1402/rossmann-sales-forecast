"""Load raw Kaggle CSVs and merge train/test with store metadata.

The single non-obvious trap here is ``StateHoliday``: depending on the CSV
version it mixes the integer ``0`` and the string ``"0"``. We normalise it to a
string column on the way in so downstream encoding is deterministic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from rossmann import config

logger = logging.getLogger(__name__)

# Explicit dtypes avoid pandas guessing StateHoliday as numeric and to keep
# memory down on the ~1M row train file.
_TRAIN_DTYPES = {
    "Store": "int32",
    "DayOfWeek": "int16",
    "Sales": "int32",
    "Customers": "int32",
    "Open": "Int8",  # nullable: test.csv has a few missing Open values
    "Promo": "int8",
    "StateHoliday": "object",
    "SchoolHoliday": "int8",
}

_STORE_DTYPES = {
    "Store": "int32",
    "StoreType": "object",
    "Assortment": "object",
    "CompetitionDistance": "float64",
    "CompetitionOpenSinceMonth": "float64",
    "CompetitionOpenSinceYear": "float64",
    "Promo2": "int8",
    "Promo2SinceWeek": "float64",
    "Promo2SinceYear": "float64",
    "PromoInterval": "object",
}


@dataclass
class RossmannData:
    """Container for the merged train and test frames."""

    train: pd.DataFrame
    test: Optional[pd.DataFrame] = None


def _read_csv(path: Path, dtypes: dict, parse_dates: Optional[list] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path}. Download the Kaggle 'Rossmann Store Sales' data into "
            f"{config.RAW_DIR} (train.csv, test.csv, store.csv)."
        )
    df = pd.read_csv(path, dtype=dtypes, parse_dates=parse_dates, low_memory=False)
    logger.info("Loaded %s: shape=%s", path.name, df.shape)
    return df


def _normalise_state_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the mixed-type StateHoliday column to a clean string in {0,a,b,c}."""
    df["StateHoliday"] = df["StateHoliday"].astype(str).str.strip()
    df.loc[df["StateHoliday"].isin(["0", "0.0", "nan"]), "StateHoliday"] = "0"
    return df


def load_raw(
    train_csv: Path = config.TRAIN_CSV,
    test_csv: Path = config.TEST_CSV,
    store_csv: Path = config.STORE_CSV,
    with_test: bool = True,
) -> RossmannData:
    """Read the three CSVs and left-merge store metadata onto train (and test)."""
    store = _read_csv(store_csv, _STORE_DTYPES)

    train = _read_csv(train_csv, _TRAIN_DTYPES, parse_dates=["Date"])
    train = _normalise_state_holiday(train)
    train = train.merge(store, on="Store", how="left")
    logger.info("Merged train+store: shape=%s", train.shape)

    test = None
    if with_test and test_csv.exists():
        test_dtypes = {k: v for k, v in _TRAIN_DTYPES.items() if k not in ("Sales", "Customers")}
        test = _read_csv(test_csv, test_dtypes, parse_dates=["Date"])
        test = _normalise_state_holiday(test)
        # Kaggle test has a handful of null Open values — treat as open (1).
        test["Open"] = test["Open"].fillna(1).astype("int8")
        test = test.merge(store, on="Store", how="left")
        logger.info("Merged test+store: shape=%s", test.shape)

    return RossmannData(train=train, test=test)
