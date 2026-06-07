"""End-to-end orchestration: load -> clean -> features -> filter -> split -> train.

The ordering here is deliberate and load-bearing:

    load + clean
        -> calendar / promo / competition features
        -> LAG features (on the full calendar, gaps intact)
        -> FILTER to open, positive-sales rows        <-- only now
        -> per-fold: StoreAggregates.fit(train) then transform(train/val)
        -> train + evaluate

Computing lags before filtering is what makes ``SalesLag7`` mean "7 calendar days
ago" rather than "the previous open day". See plan / test_features.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rossmann import config
from rossmann.data import cleaner, loader, splitter
from rossmann.evaluation import metrics
from rossmann.features import calendar, competition, lags, promotions
from rossmann.models.baseline import MedianBaseline
from rossmann.models.lgbm_model import LGBMForecaster

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Feature building
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, competition_cap: Optional[int] = None) -> pd.DataFrame:
    """Apply every stateless feature transform in the correct order.

    Lag features are added here (before filtering). Store-level aggregates are
    NOT added here — they are fold-specific and handled in ``run_cv``.
    """
    df = cleaner.clean(df)
    df = calendar.add_features(df)
    df = calendar.encode_state_holiday(df)
    df = promotions.add_features(df)
    df = competition.add_features(df, open_months_cap=competition_cap)
    df = lags.add_lag_features(df)  # MUST be before filter_trainable
    df = cleaner.add_log_target(df)
    return df


def filter_trainable(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only open days with positive sales — done AFTER lag construction.

    Mirrors Kaggle scoring (closed/zero-sales days are excluded). Lag columns
    retain their correct calendar-based values from before the filter.
    """
    before = len(df)
    out = df[(df["Open"] == 1) & (df[config.COLS.target] > 0)].copy()
    logger.info("filter_trainable: %d -> %d rows", before, len(out))
    return out


# --------------------------------------------------------------------------- #
# Cross-validated training
# --------------------------------------------------------------------------- #
def run_cv(
    df_features: pd.DataFrame,
    model_name: str = "lgbm",
    tune_on_fold1: bool = True,
) -> Dict:
    """Run 3-fold walk-forward CV; return per-fold and aggregate metrics.

    ``df_features`` must already be the output of ``build_features``. Filtering
    to trainable rows happens here so lag columns are intact upstream.
    """
    target = config.COLS.target
    results: List[Dict] = []
    tuned_params: Optional[Dict] = None
    importances: Optional[pd.Series] = None

    for fold in splitter.walk_forward_folds(df_features):
        train = filter_trainable(fold.train)
        val = filter_trainable(fold.val)

        # Fold-specific store aggregates — fitted on train only (no leakage).
        agg = lags.StoreAggregates().fit(train)
        train = agg.transform(train)
        val = agg.transform(val)

        y_train, y_val = train[target].to_numpy(), val[target].to_numpy()

        if model_name == "naive":
            from rossmann.models.baseline import NaiveLastWeek
            model = NaiveLastWeek().fit(train, y_train)
        elif model_name == "baseline":
            model = MedianBaseline().fit(train, y_train)
        elif model_name == "lgbm":
            model = LGBMForecaster()
            if tune_on_fold1 and fold.index == 1 and tuned_params is None:
                tuned_params = model.tune(train, y_train, val, y_val)
            if tuned_params:
                model.params.update(tuned_params)
            model.fit(train, y_train, X_val=val, y_val=y_val)
            importances = model.feature_importance()
        elif model_name == "mstl":
            from rossmann.models.statsforecast_model import StatsForecastMSTL

            model = StatsForecastMSTL().fit(train, y_train)
        else:
            raise ValueError(f"Unknown model '{model_name}'")

        scores = model.evaluate(val, y_val)
        scores["fold"] = fold.index
        results.append(scores)
        logger.info("%s | %s | RMSPE=%.4f", fold.label, model_name, scores["rmspe"])

    summary = _summarise(results)
    return {
        "model": model_name,
        "folds": results,
        "summary": summary,
        "tuned_params": tuned_params,
        "feature_importance": importances,
    }


def _summarise(results: List[Dict]) -> Dict[str, float]:
    arr = {m: np.array([r[m] for r in results]) for m in ("rmspe", "mae", "mape")}
    return {
        "rmspe_mean": float(arr["rmspe"].mean()),
        "rmspe_std": float(arr["rmspe"].std()),
        "mae_mean": float(arr["mae"].mean()),
        "mape_mean": float(arr["mape"].mean()),
    }


def load_and_build(with_test: bool = False) -> pd.DataFrame:
    """Convenience: load raw train, build features. Returns the feature frame."""
    data = loader.load_raw(with_test=with_test)
    return build_features(data.train)
