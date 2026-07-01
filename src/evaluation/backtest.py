"""Pseudo-out-of-sample forecast evaluation.

Recursively (expanding window) or with a rolling window, re-fit each model at every
origin using only data available up to that point, forecast h steps ahead, and score
against the realized value. Reports RMSE / MAE and the **relative RMSE** vs. the
benchmark (the standard way this literature reports results — a value < 1 means the
model beats the random walk).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..models import registry


@dataclass
class BacktestResult:
    horizon: int
    scheme: str
    forecasts: pd.DataFrame          # index=origin date, columns=model keys
    realized: pd.Series              # realized value at origin+h, indexed by origin
    metrics: pd.DataFrame            # rows=model, cols=[rmse, mae, rel_rmse, n]

    def leaderboard(self) -> pd.DataFrame:
        return self.metrics.sort_values("rel_rmse")


def _fit_and_forecast(key, y_train, X_train, h) -> Optional[float]:
    try:
        model = registry.make(key)
        model.fit(y_train, X_train)
        return model.forecast(h)
    except Exception:
        return np.nan


def run_backtest(
    y: pd.Series,
    X: Optional[pd.DataFrame],
    model_keys: List[str],
    horizon: int = 12,
    scheme: str = "expanding",       # 'expanding' | 'rolling'
    min_train: int = 120,
    rolling_window: int = 240,
    step: int = 1,
    benchmark_key: str = registry.BENCHMARK_KEY,
    progress=None,
) -> BacktestResult:
    """Backtest `model_keys` on inflation `y` (+ activity `X`) at a single horizon."""
    y = y.astype(float).dropna()
    idx = y.index
    n = len(y)

    origins = list(range(min_train, n - horizon, step))
    fc = {k: [] for k in model_keys}
    realized, origin_dates = [], []

    total = len(origins)
    for i, t in enumerate(origins):
        if scheme == "rolling":
            start = max(0, t - rolling_window)
        else:
            start = 0
        y_train = y.iloc[start:t]
        # pass full X; models align it to y_train's dates by label (trailing only,
        # so there is no look-ahead). Positional slicing would misalign the indices.
        X_train = X if X is not None else None

        for k in model_keys:
            fc[k].append(_fit_and_forecast(k, y_train, X_train, horizon))

        realized.append(float(y.iloc[t + horizon - 1]))
        origin_dates.append(idx[t])

        if progress is not None and total:
            progress((i + 1) / total)

    forecasts = pd.DataFrame(fc, index=pd.Index(origin_dates, name="origin"))
    realized_s = pd.Series(realized, index=forecasts.index, name="realized")

    metrics = _score(forecasts, realized_s, benchmark_key)
    return BacktestResult(horizon, scheme, forecasts, realized_s, metrics)


def _score(forecasts: pd.DataFrame, realized: pd.Series, benchmark_key: str) -> pd.DataFrame:
    rows = {}
    bench_rmse = None
    if benchmark_key in forecasts.columns:
        e = forecasts[benchmark_key] - realized
        bench_rmse = float(np.sqrt(np.nanmean(e**2)))

    for k in forecasts.columns:
        err = (forecasts[k] - realized).values
        valid = ~np.isnan(err)
        if valid.sum() == 0:
            rows[k] = dict(rmse=np.nan, mae=np.nan, rel_rmse=np.nan, n=0)
            continue
        rmse = float(np.sqrt(np.nanmean(err[valid] ** 2)))
        mae = float(np.nanmean(np.abs(err[valid])))
        rel = rmse / bench_rmse if bench_rmse else np.nan
        rows[k] = dict(rmse=rmse, mae=mae, rel_rmse=rel, n=int(valid.sum()))

    m = pd.DataFrame(rows).T
    m.index.name = "model"
    return m
