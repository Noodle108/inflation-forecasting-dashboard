"""Pseudo-out-of-sample forecast evaluation.

Recursively (expanding window) or with a rolling window, re-fit each model at every
origin using only data available up to that point, forecast h steps ahead, and score
against the realized value. Reports RMSE / MAE and the **relative RMSE** vs. the
benchmark (the standard way this literature reports results — a value < 1 means the
model beats the random walk).

Parallelism
-----------
When ``n_workers > 1`` (the default picks ``min(8, cpu_count - 1)``), origins are
distributed across a ``concurrent.futures.ProcessPoolExecutor``. Each worker
process keeps its own FRED / derived-panel caches in memory, so subsequent
origins on the same worker reuse the cached data. Same trick as the FW horse
race — see ``src/evaluation/fw_horserace.py``.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    realized: pd.Series              # realized value at target date, indexed by origin
    metrics: pd.DataFrame            # rows=model, cols=[rmse, mae, rel_rmse, n]
    target_dates: pd.DatetimeIndex = None  # target date of each forecast (= origin + h)

    def leaderboard(self) -> pd.DataFrame:
        return self.metrics.sort_values("rel_rmse")

    def by_target(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return (forecasts, realized) re-indexed by the target date so a chart of
        both series shows the forecast and the realized value at the same point in time.
        """
        idx = self.target_dates if self.target_dates is not None else self.forecasts.index
        fc = self.forecasts.copy()
        fc.index = idx
        r = self.realized.copy()
        r.index = idx
        return fc, r


def _fit_and_forecast(key, y_train, X_train, h) -> Optional[float]:
    try:
        model = registry.make(key)
        model.fit(y_train, X_train)
        return model.forecast(h)
    except Exception:
        return np.nan


def _origin_task(args):
    """Worker function: fit all keys on one training window at one horizon.

    Runs in a separate process. Returns a dict {key: forecast}.
    """
    y_train, X, model_keys, horizon = args
    out = {}
    for k in model_keys:
        out[k] = _fit_and_forecast(k, y_train, X, horizon)
    return out


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
    n_workers: int | None = None,
    progress=None,
) -> BacktestResult:
    """Backtest `model_keys` on inflation `y` (+ activity `X`) at a single horizon.

    When ``n_workers > 1`` origins are distributed across a process pool. Pass 1
    to force serial (useful for debugging or when the pool overhead exceeds
    the fit time).
    """
    y = y.astype(float).dropna()
    idx = y.index
    n = len(y)

    origins = list(range(min_train, n - horizon, step))
    fc = {k: [None] * len(origins) for k in model_keys}
    realized, origin_dates, target_dates = [], [], []

    if n_workers is None:
        n_workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    # Pre-compute origin-level bookkeeping (realized, dates) — needs no worker.
    tasks = []
    for i, t in enumerate(origins):
        if scheme == "rolling":
            start = max(0, t - rolling_window)
        else:
            start = 0
        y_train = y.iloc[start:t]
        target_i = t + horizon - 1
        realized.append(float(y.iloc[target_i]))
        origin_dates.append(idx[t])
        target_dates.append(idx[target_i])
        tasks.append((i, y_train))

    total = len(origins)

    def _record(i, per_key):
        for k, v in per_key.items():
            fc[k][i] = v

    if n_workers <= 1:
        # Serial path — same code, no process overhead.
        for done, (i, y_train) in enumerate(tasks, start=1):
            per_key = _origin_task((y_train, X, model_keys, horizon))
            _record(i, per_key)
            if progress is not None:
                progress(done / total)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for (i, y_train) in tasks:
                fut = pool.submit(_origin_task,
                                  (y_train, X, model_keys, horizon))
                futures[fut] = i
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    per_key = fut.result()
                except Exception:
                    per_key = {k: float("nan") for k in model_keys}
                _record(i, per_key)
                done += 1
                if progress is not None:
                    progress(done / total)

    forecasts = pd.DataFrame(fc, index=pd.Index(origin_dates, name="origin"))
    realized_s = pd.Series(realized, index=forecasts.index, name="realized")
    targets = pd.DatetimeIndex(target_dates, name="target")

    metrics = _score(forecasts, realized_s, benchmark_key)
    return BacktestResult(horizon, scheme, forecasts, realized_s, metrics,
                          target_dates=targets)


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
