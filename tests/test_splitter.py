"""Leakage guards on the walk-forward splitter."""
from __future__ import annotations

import pandas as pd

from rossmann.data import splitter
from rossmann.features import lags


def test_train_strictly_precedes_val_every_fold(synthetic_raw):
    splits = [
        ("2014-06-30", "2014-07-01", "2014-08-15"),
        ("2014-09-30", "2014-10-01", "2014-11-15"),
    ]
    folds = list(splitter.walk_forward_folds(synthetic_raw, splits=splits))
    assert len(folds) == 2
    for fold in folds:
        assert fold.train["Date"].max() < fold.val["Date"].min(), fold.label


def test_store_aggregates_fit_on_train_only(synthetic_raw):
    """Aggregates fitted on the train fold must not see validation-period sales."""
    splits = [("2014-06-30", "2014-07-01", "2014-08-15")]
    fold = next(splitter.walk_forward_folds(synthetic_raw, splits=splits))

    agg = lags.StoreAggregates().fit(fold.train)
    # The learned store mean should equal the mean computed on train only,
    # i.e. it must differ from the mean over the full (train+val) data.
    train_mean = fold.train.groupby("Store")["Sales"].mean()
    out = agg.transform(fold.train.head(1))
    store0 = fold.train.iloc[0]["Store"]
    assert out["StoreSalesMean"].iloc[0] == train_mean.loc[store0].astype("float32")


def test_holdout_split_six_weeks(synthetic_raw):
    fold = splitter.holdout_split(synthetic_raw, val_weeks=6)
    span = (fold.val["Date"].max() - fold.val["Date"].min()).days
    assert span <= 6 * 7
    assert fold.train["Date"].max() < fold.val["Date"].min()
