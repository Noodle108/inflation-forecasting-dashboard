"""Faust–Wright horse-race evaluation.

Rebuilds FW (2013) Table 1.2: for each model and each horizon h in a set (default
0..8 quarters), fit the model recursively at every origin using data available at
that origin only, forecast at horizon h, and score by root-mean-squared prediction
error (RMSPE). Report RMSPE and the ratio to a chosen benchmark (FW use the "Fixed ρ"
gap AR(1) as the divisor for their relative-RMSPE columns).

Efficiency
----------
* **Each model fits once per origin**, then is queried at every requested horizon
  (rather than re-fitting per horizon).
* When ``n_workers > 1``, origins are distributed across a process pool
  (``concurrent.futures.ProcessPoolExecutor``). Cached FRED fetches live in
  worker-process memory and are populated once per worker on first use.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ..models import registry


@dataclass
class FWResult:
    horizons: list[int]
    forecasts: dict           # {(model_key, horizon): pd.Series indexed by origin}
    realized: dict            # {horizon: pd.Series of realized value at origin+h-1}
    rmspe: pd.DataFrame       # rows=model, cols=horizons, values=RMSPE
    rel_rmspe: pd.DataFrame   # rmspe divided by benchmark row
    n: pd.DataFrame           # rows=model, cols=horizons, values=# valid pairs
    benchmark_key: str


def _fit_and_paths(key, y_train, X_train, horizons) -> list[float]:
    """Fit `key` on y_train and return the h-step forecast for every h in horizons.

    FW's convention (Table 1.2): with data through quarter t-1, column h is the
    forecast for π_{t+h}, i.e. (h+1)-steps-ahead of the last training obs. h=0
    is the nowcast of π_t. Realized target at index t+h is set by the caller.
    """
    try:
        m = registry.make(key)
        m.fit(y_train, X_train)
        return [float(m.forecast(h + 1)) for h in horizons]
    except Exception:
        return [float("nan")] * len(horizons)


def _origin_task(args):
    """Worker function: fit all requested models on one training window and
    return their forecast paths. Runs in a separate process."""
    y_train, X, keys, horizons = args
    out = {}
    for k in keys:
        out[k] = _fit_and_paths(k, y_train, X, horizons)
    return out


def run_fw_horserace(
    y: pd.Series,
    X: Optional[pd.DataFrame],
    model_keys: list[str],
    horizons: list[int],
    benchmark_key: str,
    min_train: int = 60,
    step: int = 1,
    n_workers: int | None = None,
    progress=None,
    training_by_origin: Optional[dict] = None,
) -> FWResult:
    """Score `model_keys` at every h in `horizons` on inflation series `y`.

    Each model fits once per origin. Column h is FW's h-quarter-ahead forecast:
    with training data through index t-1, the forecast targets y.iloc[t+h]
    (h=0 → nowcast of π_t, h=k → π_{t+k}). Origins iterate from `min_train`
    to `len(y) - max(horizons) - 1`.

    Real-time / vintage mode: pass ``training_by_origin`` = a dict keyed by
    origin timestamp with values ``(y_train, X_train)``. When set, models fit
    on the vintage-appropriate series for that origin instead of ``y.iloc[:t]``
    / ``X.iloc[:t]``. Realized targets always come from the current-vintage
    ``y`` (that IS the final revised value, which is what we score against —
    same convention Faust–Wright use).

    Parallelism: when ``n_workers`` is None (default), pick a sensible size
    based on cpu_count. Pass 1 to run serially.
    """
    y = y.astype(float).dropna()
    n_obs = len(y)
    h_max = max(horizons)

    # y_train ends at index t-1; target for column h is y.iloc[t+h] (FW convention).
    # Largest origin t satisfies t + h_max <= n_obs - 1.
    origins = list(range(min_train, n_obs - h_max, step))
    forecasts = {(k, h): [None] * len(origins) for k in model_keys for h in horizons}
    realized = {h: [] for h in horizons}
    origin_dates = []

    if n_workers is None:
        n_workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    # Build per-origin task tuples and precompute the realized column
    tasks = []
    for i, t in enumerate(origins):
        for h in horizons:
            idx = t + h
            realized[h].append(float(y.iloc[idx]) if idx < n_obs else float("nan"))
        origin_ts = y.index[t]
        origin_dates.append(origin_ts)
        if training_by_origin is not None and origin_ts in training_by_origin:
            y_train, X_train = training_by_origin[origin_ts]
        else:
            y_train, X_train = y.iloc[:t], X
        tasks.append((i, y_train, X_train))

    total = len(origins)

    def _record(i, per_key_paths):
        for k, paths in per_key_paths.items():
            for path_val, h in zip(paths, horizons):
                forecasts[(k, h)][i] = path_val

    if n_workers <= 1:
        # Serial path — same code, no process overhead. Useful for debugging.
        for done, (i, y_train, X_train) in enumerate(tasks, start=1):
            per_key = _origin_task((y_train, X_train, model_keys, horizons))
            _record(i, per_key)
            if progress is not None:
                progress(done / total)
    else:
        # Distribute origins across worker processes. Each worker keeps its
        # own FRED cache and derived-panel cache in memory, so subsequent
        # origins on the same worker are much cheaper than the first.
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for (i, y_train, X_train) in tasks:
                fut = pool.submit(_origin_task,
                                  (y_train, X_train, model_keys, horizons))
                futures[fut] = i
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    per_key = fut.result()
                except Exception:
                    per_key = {k: [float("nan")] * len(horizons) for k in model_keys}
                _record(i, per_key)
                done += 1
                if progress is not None:
                    progress(done / total)

    # Convert to Series/DataFrames
    idx_o = pd.Index(origin_dates, name="origin")
    fc_series = {(k, h): pd.Series(v, index=idx_o) for (k, h), v in forecasts.items()}
    real_series = {h: pd.Series(v, index=idx_o) for h, v in realized.items()}

    # Score
    rmspe = pd.DataFrame(index=model_keys, columns=horizons, dtype=float)
    nvalid = pd.DataFrame(index=model_keys, columns=horizons, dtype=float)
    for k in model_keys:
        for h in horizons:
            err = (fc_series[(k, h)] - real_series[h]).values
            valid = ~np.isnan(err)
            if valid.sum() == 0:
                rmspe.at[k, h] = np.nan
                nvalid.at[k, h] = 0
                continue
            rmspe.at[k, h] = float(np.sqrt(np.mean(err[valid] ** 2)))
            nvalid.at[k, h] = int(valid.sum())

    if benchmark_key in rmspe.index:
        rel = rmspe.divide(rmspe.loc[benchmark_key], axis=1)
    else:
        rel = rmspe.copy()

    return FWResult(horizons=list(horizons), forecasts=fc_series,
                    realized=real_series, rmspe=rmspe, rel_rmspe=rel, n=nvalid,
                    benchmark_key=benchmark_key)
