#!/usr/bin/env python
"""Print a per-fold and per-segment metric breakdown for a model.

Re-runs CV (cheap for baseline; for lgbm pass --no-tune to skip the sweep) and
reports RMSPE broken down by StoreType and DayOfWeek — useful for spotting where
the model is weak (e.g. Sundays, Type-b stores).

    python scripts/evaluate.py --model baseline
    python scripts/evaluate.py --model lgbm --no-tune
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rossmann import config, pipeline  # noqa: E402
from rossmann.data import splitter  # noqa: E402
from rossmann.evaluation import metrics  # noqa: E402
from rossmann.features import lags  # noqa: E402
from rossmann.models.baseline import MedianBaseline  # noqa: E402
from rossmann.models.lgbm_model import LGBMForecaster  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")


def _segment_rmspe(df: pd.DataFrame, by: str) -> pd.Series:
    return df.groupby(by).apply(
        lambda g: metrics.rmspe(g["Sales"], g["pred"]), include_groups=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment-level evaluation.")
    parser.add_argument("--model", choices=["lgbm", "baseline"], default="baseline")
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()

    feats = pipeline.load_and_build(with_test=False)

    # Evaluate on the last fold (most recent regime) for the segment breakdown.
    last_fold = list(splitter.walk_forward_folds(feats))[-1]
    train = pipeline.filter_trainable(last_fold.train)
    val = pipeline.filter_trainable(last_fold.val)

    agg = lags.StoreAggregates().fit(train)
    train, val = agg.transform(train), agg.transform(val)
    y_train = train["Sales"].to_numpy()

    if args.model == "baseline":
        model = MedianBaseline().fit(train, y_train)
    else:
        model = LGBMForecaster()
        model.fit(train, y_train, X_val=val, y_val=val["Sales"].to_numpy())

    val = val.copy()
    val["pred"] = model.predict(val)

    overall = metrics.evaluate(val["Sales"], val["pred"])
    print(f"\n=== {args.model} | last fold ({last_fold.label}) ===")
    print(f"Overall  RMSPE={overall['rmspe']:.4f}  MAE={overall['mae']:.1f}  "
          f"MAPE={overall['mape']:.4f}\n")
    print("RMSPE by StoreType:\n", _segment_rmspe(val, "StoreType").round(4), "\n")
    print("RMSPE by DayOfWeek:\n", _segment_rmspe(val, "DayOfWeek").round(4))


if __name__ == "__main__":
    main()
