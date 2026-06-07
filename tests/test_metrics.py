"""RMSPE / MAE / MAPE edge cases. These pin the metric before any model exists."""
from __future__ import annotations

import numpy as np
import pytest

from rossmann.evaluation import metrics


def test_perfect_prediction_is_zero():
    y = np.array([100.0, 200.0, 300.0])
    assert metrics.rmspe(y, y) == pytest.approx(0.0)
    assert metrics.mae(y, y) == pytest.approx(0.0)
    assert metrics.mape(y, y) == pytest.approx(0.0)


def test_known_ten_percent_error():
    # Every prediction is 10% low => each percentage error is 0.1 => RMSPE = 0.1.
    y_true = np.array([100.0, 200.0, 400.0])
    y_pred = y_true * 0.9
    assert metrics.rmspe(y_true, y_pred) == pytest.approx(0.1)
    assert metrics.mape(y_true, y_pred) == pytest.approx(0.1)


def test_zero_sales_days_are_excluded():
    # The zero-true entry must not blow up RMSPE (no division by zero).
    y_true = np.array([0.0, 100.0, 100.0])
    y_pred = np.array([50.0, 90.0, 110.0])
    # Only the two non-zero entries count: errors of -0.1 and +0.1 => RMSPE 0.1.
    assert metrics.rmspe(y_true, y_pred) == pytest.approx(0.1)
    assert np.isfinite(metrics.rmspe(y_true, y_pred))


def test_all_zero_true_returns_zero_not_nan():
    y_true = np.zeros(3)
    y_pred = np.array([1.0, 2.0, 3.0])
    assert metrics.rmspe(y_true, y_pred) == 0.0


def test_single_element():
    assert metrics.rmspe([100.0], [80.0]) == pytest.approx(0.2)


def test_mae_simple():
    assert metrics.mae([10.0, 20.0], [12.0, 18.0]) == pytest.approx(2.0)


def test_evaluate_bundle_keys():
    out = metrics.evaluate([100.0, 200.0], [90.0, 180.0])
    assert set(out) == {"rmspe", "mae", "mape"}
    assert out["rmspe"] == pytest.approx(0.1)


def test_lgbm_rmspe_eval_matches_rmspe():
    class _DS:  # minimal stand-in for a lightgbm Dataset
        def __init__(self, label):
            self._label = label

        def get_label(self):
            return self._label

    y_true = np.array([100.0, 200.0, 400.0])
    y_pred = y_true * 0.9
    ds = _DS(np.log1p(y_true))
    name, value, higher_better = metrics.lgbm_rmspe_eval(np.log1p(y_pred), ds)
    assert name == "rmspe"
    assert higher_better is False
    assert value == pytest.approx(0.1, abs=1e-6)
