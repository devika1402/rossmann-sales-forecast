"""Smoke test: a tiny synthetic frame through features -> filter -> CV."""
from __future__ import annotations

import numpy as np

from rossmann import pipeline
from rossmann.features import lags


def test_build_features_then_filter(synthetic_raw):
    feats = pipeline.build_features(synthetic_raw)
    # Lag column present and built on the full calendar (some early NaNs expected).
    assert "SalesLag7" in feats.columns
    assert feats["SalesLag7"].isna().any()

    filtered = pipeline.filter_trainable(feats)
    # Closed (Sunday) rows and zero-sales rows are gone.
    assert (filtered["Open"] == 1).all()
    assert (filtered["Sales"] > 0).all()
    # Filtering must not have reordered away the lag values.
    assert "SalesLag7" in filtered.columns


def test_store_aggregates_have_customers_proxy(synthetic_raw):
    feats = pipeline.build_features(synthetic_raw)
    filtered = pipeline.filter_trainable(feats)
    agg = lags.StoreAggregates().fit(filtered)
    out = agg.transform(filtered)
    assert "StoreDowAvgCustomers" in out.columns
    assert out["StoreDowAvgCustomers"].notna().all()


def test_baseline_cv_runs_and_scores(synthetic_raw):
    feats = pipeline.build_features(synthetic_raw)
    # Use small custom folds that fit inside the synthetic date range.
    import rossmann.config as cfg

    folds = [
        ("2014-08-31", "2014-09-01", "2014-10-15"),
        ("2014-10-31", "2014-11-01", "2014-12-15"),
    ]
    original = cfg.SPLITS.folds
    object.__setattr__(cfg.SPLITS, "folds", folds)
    try:
        out = pipeline.run_cv(feats, model_name="baseline")
    finally:
        object.__setattr__(cfg.SPLITS, "folds", original)

    assert len(out["folds"]) == 2
    assert np.isfinite(out["summary"]["rmspe_mean"])
    assert out["summary"]["rmspe_mean"] >= 0
