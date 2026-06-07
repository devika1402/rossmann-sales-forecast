"""LightGBM forecaster trained on log1p(Sales) with an Optuna sweep on Fold 1.

Design choices (see README / plan):
* Target is ``log1p(Sales)``; predictions are ``expm1``-ed back to real units.
  Training MSE on the log target approximates the scale-invariant RMSPE metric.
* Low-cardinality categoricals are passed to LightGBM natively — no one-hot.
* Hyperparameters are tuned for real (30-trial Optuna sweep), not a commented stub.
  ``tune`` runs the sweep on a single fold; ``fit`` then trains with fixed params.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from rossmann import config
from rossmann.evaluation import metrics
from rossmann.models.base import BaseForecaster

logger = logging.getLogger(__name__)


class LGBMForecaster(BaseForecaster):
    """Gradient-boosted trees on the log target."""

    name = "lgbm"

    def __init__(
        self,
        params: Optional[Dict] = None,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> None:
        self.cfg = config.LGBM
        self.params = dict(params) if params else dict(self.cfg.params)
        self.feature_cols = feature_cols
        self.categorical_cols = categorical_cols or list(config.COLS.categorical)
        self.booster: Optional[lgb.Booster] = None
        self.best_iteration_: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self.feature_cols or [c for c in X.columns if c not in _NON_FEATURES]
        self.feature_cols = cols
        Xm = X[cols].copy()
        for c in self.categorical_cols:
            if c in Xm.columns:
                Xm[c] = Xm[c].astype("category")
        return Xm

    def _base_params(self) -> Dict:
        return {
            "objective": self.cfg.objective,
            "metric": self.cfg.metric,
            "learning_rate": self.cfg.learning_rate,
            "num_leaves": self.params["num_leaves"],
            "min_child_samples": self.params["min_child_samples"],
            "feature_fraction": self.params["feature_fraction"],
            "bagging_fraction": self.params["bagging_fraction"],
            "bagging_freq": self.params.get("bagging_freq", 1),
            "seed": self.cfg.random_state,
            "num_threads": self.cfg.n_jobs,
            "verbosity": self.cfg.verbosity,
        }

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LGBMForecaster":
        """Train on real-sales ``y`` (converted to log internally).

        If a validation set is given, RMSPE-based early stopping is used.
        """
        Xm = self._matrix(X)
        y_log = np.log1p(np.asarray(y, dtype="float64"))
        train_set = lgb.Dataset(Xm, label=y_log, categorical_feature=self.categorical_cols)

        valid_sets, callbacks = [train_set], []
        if X_val is not None and y_val is not None:
            Xv = self._matrix(X_val)
            val_set = lgb.Dataset(
                Xv, label=np.log1p(np.asarray(y_val, dtype="float64")), reference=train_set
            )
            valid_sets = [val_set]
            callbacks = [
                lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=200),
            ]

        self.booster = lgb.train(
            self._base_params(),
            train_set,
            num_boost_round=self.cfg.n_estimators,
            valid_sets=valid_sets,
            feval=metrics.lgbm_rmspe_eval,
            callbacks=callbacks,
        )
        self.best_iteration_ = self.booster.best_iteration or self.cfg.n_estimators
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("LGBMForecaster.predict called before fit().")
        Xm = self._matrix(X)
        preds_log = self.booster.predict(Xm, num_iteration=self.best_iteration_)
        return np.expm1(preds_log).clip(min=0)

    def feature_importance(self) -> pd.Series:
        """Gain-based importance, sorted descending."""
        gains = self.booster.feature_importance(importance_type="gain")
        return pd.Series(gains, index=self.feature_cols).sort_values(ascending=False)

    # ------------------------------------------------------------------ #
    # Optuna tuning (runs on a single fold, e.g. Fold 1)
    # ------------------------------------------------------------------ #
    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: Optional[int] = None,
    ) -> Dict:
        """Search {num_leaves, min_child_samples, feature/bagging fraction} by RMSPE.

        Returns the best params and stores them on ``self.params`` for later ``fit``.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = n_trials or self.cfg.n_optuna_trials

        Xm = self._matrix(X)
        Xv = self._matrix(X_val)
        y_log = np.log1p(np.asarray(y, dtype="float64"))
        y_val_real = np.asarray(y_val, dtype="float64")
        # feature_pre_filter=False: the dataset is reused across trials with
        # different min_child_samples, which would otherwise raise.
        train_set = lgb.Dataset(
            Xm,
            label=y_log,
            categorical_feature=self.categorical_cols,
            params={"feature_pre_filter": False},
        )

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "objective": self.cfg.objective,
                "metric": self.cfg.metric,
                "learning_rate": self.cfg.learning_rate,
                "num_leaves": trial.suggest_int("num_leaves", 31, 511),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": 1,
                "seed": self.cfg.random_state,
                "num_threads": self.cfg.n_jobs,
                "verbosity": self.cfg.verbosity,
            }
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=self.cfg.n_estimators,
                valid_sets=[lgb.Dataset(Xv, label=np.log1p(y_val_real), reference=train_set)],
                feval=metrics.lgbm_rmspe_eval,
                callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)],
            )
            preds = np.expm1(booster.predict(Xv, num_iteration=booster.best_iteration))
            return metrics.rmspe(y_val_real, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        logger.info("Optuna best RMSPE=%.4f params=%s", study.best_value, study.best_params)
        self.params.update(study.best_params)
        return dict(study.best_params)


# Columns that are never model inputs (target, ids, raw train-only, helper cols).
# Note: StoreType / Assortment ARE features (handled as native categoricals).
_NON_FEATURES = {
    config.COLS.target,
    config.COLS.log_target,
    config.COLS.date,
    "Customers",          # train-only; proxied by StoreDowAvgCustomers
    "Open",               # always 1 after filtering
    "StateHoliday",       # raw string; use StateHolidayEnc
    "PromoInterval",      # raw string; decoded into Promo2Active
    "Promo2Months",       # helper set column
}
