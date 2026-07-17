"""Shared utilities for the Faust–Wright (2013) forecasting suite.

Faust & Wright's central device is the **local-mean trend** τ_t, around which each
forecast is written in "gap" form (g_t = π_t − τ_t). Their preferred τ_t is the most
recent Blue-Chip 5–10y long-run inflation forecast (available from 1979); pre-1979
they fall back to exponential smoothing (α = 0.95) of realized inflation (footnote 8).

Blue-Chip is not on FRED, so we anchor τ_t on the Cleveland Fed 10-year expected
inflation series (`EXPINF10YR`, Haubrich-Pennacchi-Ritchken 2012) once available
(~1982), and fall back to FW's own exponential-smoothing formula
    τ_t = α τ_{t-1} + (1 − α) π_t,     α = 0.95
for earlier dates. Both series are annualized percent, same units as our π_t, so no
scaling is needed.

Also collected here: BIC lag selection for AR-GAP, and a tiny direct h-step OLS helper
that most FW models share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ALPHA_TAU = 0.95      # FW's exponential-smoothing coefficient (footnote 8)
FIXED_RHO = 0.46      # FW's fixed-ρ benchmark AR(1) coefficient
MAX_LAGS = 4          # cap on BIC lag search for the AR-GAP order


_EXPINF10_CACHE: dict[str, pd.Series | None] = {}


def _expinf10() -> pd.Series | None:
    """Fetch Cleveland Fed 10-yr expected inflation, cached across calls.

    Returns None if no FRED key or the series can't be fetched — callers should
    then fall back to exp-smoothing.
    """
    if "s" in _EXPINF10_CACHE:
        return _EXPINF10_CACHE["s"]
    try:
        from ..data import fred as _f
        s = _f._fetch_fred_series("EXPINF10YR", "1960-01-01")
    except Exception:
        s = None
    _EXPINF10_CACHE["s"] = s
    return s


def _exp_smooth(pi: pd.Series, alpha: float) -> pd.Series:
    """τ_t = α τ_{t-1} + (1-α) π_t, seeded at π_0. FW footnote 8."""
    vals = pi.values.astype(float)
    out = np.empty(len(vals))
    out[0] = vals[0]
    for t in range(1, len(vals)):
        out[t] = alpha * out[t - 1] + (1.0 - alpha) * vals[t]
    return pd.Series(out, index=pi.index, name="tau")


def local_mean_trend(pi: pd.Series, alpha: float = ALPHA_TAU) -> pd.Series:
    """Local-mean trend τ_t: EXPINF10YR when available, α=0.95 exp-smoothing otherwise.

    Returns a Series aligned to `pi`. The two branches are stitched together at the
    earliest available EXPINF10YR date — before that we use FW's exp-smoothing seed;
    from then on, we resample EXPINF10YR to π_t's index and forward-fill.
    """
    pi = pi.astype(float).dropna()
    if len(pi) == 0:
        return pd.Series(dtype=float, name="tau")

    exp10 = _expinf10()
    if exp10 is None or exp10.empty:
        return _exp_smooth(pi, alpha)

    # Resample EXPINF10YR to pi's cadence, then reindex + ffill onto pi's dates.
    freq = (getattr(pi.index, "freqstr", None) or "").upper()
    if not freq and len(pi) > 1:
        median_days = float(np.median(np.diff(pi.index.asi8)) / 86_400e9)
        freq = "Q" if median_days > 45 else "M"
    exp10_r = exp10.resample("QS" if freq.startswith("Q") else "MS").mean()
    exp10_on_pi = exp10_r.reindex(pi.index).ffill()

    # Exp-smoothing prefix: use it up to (and including) the last date where EXPINF10YR
    # is still NaN after ffill — those are dates that predate the series entirely.
    tau_es = _exp_smooth(pi, alpha)
    tau = exp10_on_pi.copy()
    missing = tau.isna()
    if missing.any():
        tau.loc[missing] = tau_es.loc[missing]
    tau.name = "tau"
    return tau


def bic_ar_gap_lag(gap: pd.Series, max_p: int = MAX_LAGS) -> int:
    """Pick lag order p by BIC on the gap series, using an AR(p) fit."""
    import statsmodels.api as sm
    g = gap.dropna().values
    best_p, best_bic = 1, np.inf
    for p in range(1, max_p + 1):
        y = g[p:]
        X = np.column_stack([g[p - l - 1:len(g) - l - 1] for l in range(p)])
        X = sm.add_constant(X)
        try:
            res = sm.OLS(y, X).fit()
        except Exception:
            continue
        if res.bic < best_bic:
            best_p, best_bic = p, float(res.bic)
    return best_p


def direct_h_step_ols(y_target: pd.Series, X: pd.DataFrame) -> "sm.regression.linear_model.RegressionResultsWrapper":
    """OLS with an added constant. Convenience helper for direct-h-step regressions."""
    import statsmodels.api as sm
    df = pd.concat([y_target.rename("y"), X], axis=1).dropna()
    return sm.OLS(df["y"].values, sm.add_constant(df.drop(columns="y").values)).fit()
