#!/usr/bin/env python
"""Train LightGBM on all available history and write a Kaggle submission.csv.

Trains on the full feature frame (no held-out validation — we want every row for
the final model), then predicts test.csv. Closed test days are forced to 0 sales,
matching Kaggle's convention.

    python scripts/predict.py
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rossmann import config, pipeline  # noqa: E402
from rossmann.data import loader  # noqa: E402
from rossmann.features import lags  # noqa: E402
from rossmann.models.lgbm_model import LGBMForecaster  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("predict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission.csv.")
    parser.add_argument("--out", default=str(config.OUTPUTS_DIR / "submission.csv"))
    args = parser.parse_args()

    data = loader.load_raw(with_test=True)
    if data.test is None:
        raise FileNotFoundError("test.csv not found in data/raw — cannot predict.")

    # Build features on train+test together so lag history flows into test rows,
    # then split back out. Test has no Sales/Customers; those features stay NaN
    # for test (LightGBM handles NaN) or come from train-fitted aggregates.
    data.test[config.COLS.target] = np.nan
    combined = pd.concat([data.train, data.test], ignore_index=True, sort=False)
    feats = pipeline.build_features(combined)

    if "Id" not in feats.columns:
        raise RuntimeError("test.csv is missing the 'Id' column required for submission.")
    # Test rows are exactly those carrying an Id; train rows have Id = NaN.
    train_feats = pipeline.filter_trainable(feats[feats["Id"].isna()])
    test_feats = feats[feats["Id"].notna()].copy()

    # Store aggregates fitted on train only, applied to both.
    agg = lags.StoreAggregates().fit(train_feats)
    train_feats = agg.transform(train_feats)
    test_feats = agg.transform(test_feats)

    # Reuse tuned params if a prior train.py run saved them.
    params_path = config.MODELS_DIR / "lgbm_params.pkl"
    params = None
    if params_path.exists():
        with open(params_path, "rb") as fh:
            params = pickle.load(fh)
        logger.info("Loaded tuned params: %s", params)

    model = LGBMForecaster(params=params)
    y = train_feats[config.COLS.target].to_numpy()
    logger.info("Training final LightGBM on %d rows...", len(train_feats))
    model.fit(train_feats, y)

    preds = model.predict(test_feats)
    # Closed stores sell nothing.
    preds = np.where(test_feats["Open"].to_numpy() == 0, 0.0, preds)

    submission = pd.DataFrame(
        {"Id": test_feats["Id"].astype(int), "Sales": preds}
    ).sort_values("Id")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    logger.info("Wrote %s (%d rows)", args.out, len(submission))


if __name__ == "__main__":
    main()
