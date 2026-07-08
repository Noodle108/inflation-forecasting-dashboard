"""Faust–Wright horse-race evaluation.

Rebuilds FW (2013) Table 1.2: for each model and each horizon h in a set (default
0..8 quarters), fit the model recursively at every origin using data available at
that origin only, forecast at horizon h, and score by root-mean-squared prediction
error (RMSPE). Report RMSPE and the ratio to a chosen benchmark (FW use the "Fixed ρ"
gap AR(1) as the divisor for their relative-RMSPE columns).

The core efficiency trick: **each model is fit once per origin**, and then queried
at every horizon in the requested list, rather than being re-fit per horizon.
"""
from __future__ import annotations

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
    """Fit `key` on y_train and return the h-step forecast for every h in horizons."""
    try:
        m = registry.make(key)
        m.fit(y_train, X_train)
        return [float(m.forecast(h if h > 0 else 1)) if h > 0
                else float(y_train.iloc[-1])  # h=0 = nowcast; use last obs as ref
                for h in horizons]
    except Exception:
        return [float("nan")] * len(horizons)


def run_fw_horserace(
    y: pd.Series,
    X: Optional[pd.DataFrame],
    model_keys: list[str],
    horizons: list[int],
    benchmark_key: str,
    min_train: int = 60,
    step: int = 1,
    progress=None,
) -> FWResult:
    """Score `model_keys` at every h in `horizons` on inflation series `y`.

    Each model fits once per origin. For h=0 (nowcast) we use `y_train.iloc[-1]`
    as a naive nowcast reference (models can't nowcast in this reduced-form
    setup without a proper end-of-quarter data structure). Non-zero horizons
    use forecast(h). Origins iterate from `min_train` to `len(y) - max(horizons) - 1`.
    """
    y = y.astype(float).dropna()
    n_obs = len(y)
    h_max = max(horizons)

    origins = list(range(min_train, n_obs - h_max, step))
    forecasts = {(k, h): [] for k in model_keys for h in horizons}
    realized = {h: [] for h in horizons}
    origin_dates = []

    total = len(origins) * len(model_keys)
    done = 0
    for t in origins:
        y_train = y.iloc[:t]
        # Realized: value at origin + h - 1 for h≥1; value at origin itself for h=0.
        for h in horizons:
            idx = t if h == 0 else t + h - 1
            realized[h].append(float(y.iloc[idx]) if idx < n_obs else float("nan"))
        origin_dates.append(y.index[t])

        for k in model_keys:
            paths = _fit_and_paths(k, y_train, X, horizons)
            for path_val, h in zip(paths, horizons):
                forecasts[(k, h)].append(path_val)
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
