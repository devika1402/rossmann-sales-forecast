"""Classical time-series comparison model: MSTL via Nixtla's StatsForecast.

Why StatsForecast and not Prophet: Prophet was deprecated by Meta in 2024.
StatsForecast is actively maintained, 10-100x faster, and exposes MSTL (Multiple
Seasonal-Trend decomposition using Loess), which gives the same interpretable
trend + weekly + yearly story Prophet was used for.

Scope — cluster level: we fit ONE series per StoreType (a/b/c/d) on the mean
daily sales of that cluster. This is an explicit simplification (intra-type
variance is high), suitable for trend/seasonality storytelling rather than a
production store-level forecaster.

To still place MSTL in the same store-level results table as LightGBM, we
disaggregate top-down: a store's prediction is its cluster's forecast scaled by
``store_mean / cluster_mean`` (both learned on train). This is standard
hierarchical reconciliation and keeps the comparison honest.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from rossmann import config
from rossmann.models.base import BaseForecaster

logger = logging.getLogger(__name__)

_STORE = config.COLS.store
_DATE = config.COLS.date
_TARGET = config.COLS.target


class StatsForecastMSTL(BaseForecaster):
    """Cluster-level MSTL with top-down disaggregation to stores."""

    name = "mstl"

    def __init__(self, season_length: Optional[list] = None) -> None:
        self.season_length = season_length  # resolved per-series in fit (history-aware)
        self._sf = None
        self._forecast: Optional[pd.DataFrame] = None
        self._store_to_type: Dict[int, str] = {}
        self._store_scale: Dict[int, float] = {}
        self._train_end: Optional[pd.Timestamp] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StatsForecastMSTL":
        from statsforecast import StatsForecast
        from statsforecast.models import MSTL, AutoETS

        df = X[[_STORE, "StoreType", _DATE]].copy()
        df[_TARGET] = np.asarray(y, dtype="float64")
        self._train_end = df[_DATE].max()

        # Store -> type map and per-store top-down scaling factors.
        self._store_to_type = df.groupby(_STORE)["StoreType"].first().to_dict()
        store_mean = df.groupby(_STORE)[_TARGET].mean()
        type_mean = df.groupby("StoreType")[_TARGET].mean()
        for store, s_mean in store_mean.items():
            t_mean = type_mean.get(self._store_to_type[store], s_mean)
            self._store_scale[store] = float(s_mean / t_mean) if t_mean else 1.0

        # Long panel: one series per StoreType (mean daily sales of the cluster).
        panel = (
            df.groupby(["StoreType", _DATE])[_TARGET]
            .mean()
            .reset_index()
            .rename(columns={"StoreType": "unique_id", _DATE: "ds", _TARGET: "y"})
            .sort_values(["unique_id", "ds"])
        )

        # MSTL needs >= 2 full periods of the longest season; fall back to weekly.
        hist_len = panel.groupby("unique_id").size().min()
        seasons = self.season_length or ([7, 365] if hist_len >= 730 else [7])
        logger.info("MSTL season_length=%s (min cluster history=%d days)", seasons, hist_len)

        # Trend forecaster must be non-seasonal (ZZN) — MSTL handles seasonality.
        models = [MSTL(season_length=seasons, trend_forecaster=AutoETS(model="ZZN"))]
        self._sf = StatsForecast(models=models, freq="D", n_jobs=1)
        self._sf.fit(panel)
        self._fitted_panel = panel
        return self

    def _ensure_forecast(self, horizon: int) -> pd.DataFrame:
        if self._forecast is None or self._forecast_horizon < horizon:
            self._forecast = self._sf.predict(h=horizon).reset_index()
            self._forecast_horizon = horizon
        return self._forecast

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._sf is None:
            raise RuntimeError("StatsForecastMSTL.predict called before fit().")
        horizon = max(1, (X[_DATE].max() - self._train_end).days)
        fc = self._ensure_forecast(horizon)

        # Lookup: (type, date) -> cluster forecast.
        fc_map = fc.set_index(["unique_id", "ds"])["MSTL"]
        types = X[_STORE].map(self._store_to_type).to_numpy()
        dates = X[_DATE].to_numpy()
        keys = list(zip(types, pd.to_datetime(dates)))
        cluster_pred = fc_map.reindex(keys).to_numpy()

        scales = X[_STORE].map(self._store_scale).fillna(1.0).to_numpy()
        preds = cluster_pred * scales
        return np.nan_to_num(preds, nan=0.0).clip(min=0)

    def decomposition(self) -> Dict[str, pd.DataFrame]:
        """Return per-cluster MSTL components (trend/seasonal) for plotting.

        Used by notebook 03 to show the interpretable decomposition per store type.
        """
        if self._sf is None:
            raise RuntimeError("decomposition() called before fit().")
        components: Dict[str, pd.DataFrame] = {}
        # StatsForecast stores fitted MSTL objects; each exposes its decomposition.
        for uid, fitted_model in zip(self._sf.uids, self._sf.fitted_[:, 0]):
            mstl_obj = getattr(fitted_model, "model_", None)
            if isinstance(mstl_obj, dict) and "trend" in mstl_obj:
                comp = pd.DataFrame({k: v for k, v in mstl_obj.items() if np.ndim(v) == 1})
                components[str(uid)] = comp
        return components
