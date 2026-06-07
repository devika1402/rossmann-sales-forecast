"""Central configuration: paths, column groups, and model hyperparameters.

Everything that another module might want to tweak lives here so the rest of the
codebase never hard-codes a path or a magic number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
STORE_CSV = RAW_DIR / "store.csv"


@dataclass(frozen=True)
class Columns:
    """Named column groups, so feature/model code references intent not strings."""

    target: str = "Sales"
    log_target: str = "SalesLog"
    date: str = "Date"
    store: str = "Store"

    # Categorical features handed to LightGBM natively (no one-hot).
    categorical: List[str] = field(
        default_factory=lambda: [
            "StoreType",
            "Assortment",
            "DayOfWeek",
            "Month",
            "StateHolidayEnc",
        ]
    )

    # Columns present in train.csv but NOT in test.csv — never use raw as features.
    train_only: List[str] = field(default_factory=lambda: ["Customers"])


@dataclass(frozen=True)
class LGBMConfig:
    """LightGBM defaults. The *_search ranges drive the Optuna sweep on Fold 1."""

    # Fixed across all trials.
    objective: str = "regression"
    metric: str = "rmse"  # on log-target; RMSPE is reported via custom callback
    n_estimators: int = 3000
    learning_rate: float = 0.03
    early_stopping_rounds: int = 100
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = -1

    # Sensible starting params; overwritten by Optuna's best on Fold 1.
    params: Dict = field(
        default_factory=lambda: {
            "num_leaves": 128,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
        }
    )

    n_optuna_trials: int = 30


@dataclass(frozen=True)
class SplitConfig:
    """3-fold expanding-window walk-forward validation.

    Each tuple is (train_end, val_start, val_end) — validation always immediately
    follows training with no gap, mimicking the 6-week Kaggle test horizon.
    """

    folds: List[tuple] = field(
        default_factory=lambda: [
            ("2014-12-31", "2015-01-01", "2015-03-31"),  # Fold 1 — Optuna runs here
            ("2015-03-31", "2015-04-01", "2015-06-15"),  # Fold 2
            ("2015-06-15", "2015-06-16", "2015-07-31"),  # Fold 3
        ]
    )


# Singletons imported elsewhere.
COLS = Columns()
LGBM = LGBMConfig()
SPLITS = SplitConfig()

# Imputation constants (documented where used).
NO_COMPETITOR_DISTANCE = 100_000.0  # metres; null distance => no nearby competitor
HAS_COMPETITOR_THRESHOLD = 1_000.0  # metres
