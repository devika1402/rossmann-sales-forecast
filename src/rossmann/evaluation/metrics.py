"""Forecast error metrics. RMSPE is the official Kaggle Rossmann metric.

All functions operate on real (non-log) sales. Zero-sales days are excluded from
percentage metrics because the percentage error is undefined there — this matches
Kaggle's own scoring, which ignores closed-store days.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype="float64").ravel()


def rmspe(y_true, y_pred) -> float:
    """Root Mean Squared Percentage Error (Kaggle Rossmann metric).

    RMSPE = sqrt( mean( ((y_true - y_pred) / y_true)^2 ) ), over y_true != 0.
    """
    y_true, y_pred = _as_array(y_true), _as_array(y_pred)
    mask = y_true != 0
    if not mask.any():
        return 0.0
    pct_err = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(pct_err**2)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = _as_array(y_true), _as_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (fraction, not %), over y_true != 0."""
    y_true, y_pred = _as_array(y_true), _as_array(y_pred)
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def evaluate(y_true, y_pred) -> Dict[str, float]:
    """Return all three metrics as a dict — the standard reporting bundle."""
    return {"rmspe": rmspe(y_true, y_pred), "mae": mae(y_true, y_pred), "mape": mape(y_true, y_pred)}


def lgbm_rmspe_eval(y_pred: np.ndarray, dataset) -> tuple:
    """LightGBM custom eval: RMSPE in real units, given a log1p-trained model.

    Signature matches LightGBM's ``feval`` callback. ``dataset`` carries the
    log-space labels; we expm1 both sides before scoring.
    Returns ``(eval_name, value, is_higher_better)``.
    """
    y_true = np.expm1(dataset.get_label())
    y_hat = np.expm1(y_pred)
    return "rmspe", rmspe(y_true, y_hat), False
