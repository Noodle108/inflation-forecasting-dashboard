"""Shared utilities for the Faust–Wright (2013) forecasting suite.

Faust & Wright's central device is the **local-mean trend** τ_t, around which each
forecast is written in "gap" form (g_t = π_t − τ_t). Their preferred τ_t is the most
recent Blue-Chip 5–10y long-run inflation forecast (available from 1979); pre-1979
they fall back to exponential smoothing (α = 0.95) of realized inflation (footnote 8).

Blue-Chip is subscription-only. We build τ_t from three ordered fallbacks:

    1. **SPF 10-year CPI** (Philly Fed sheet ``CPI10``, 1991Q4→). Same object as
       Blue Chip — a quarterly consensus of professional forecasters at a 5–10y
       long horizon — just published by the Philly Fed rather than Wolters Kluwer.
       This is the standard academic replacement when Blue Chip isn't available.
    2. **Cleveland Fed 10-yr expected inflation** (FRED ``EXPINF10YR``, 1982→).
       Model-based (Haubrich–Pennacchi–Ritchken 2012) — used when SPF is missing.
    3. **FW's exponential-smoothing fallback** (footnote 8):
           τ_t = α τ_{t-1} + (1 − α) π_t,     α = 0.95
       Used for dates before both survey/model anchors are available.

All three anchors are annualized percent — same units as π_t, no scaling needed.

Also collected here: BIC lag selection for AR-GAP, and a tiny direct h-step OLS helper
that most FW models share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ALPHA_TAU = 0.95      # FW's exponential-smoothing coefficient (footnote 8)
FIXED_RHO = 0.46      # FW's fixed-ρ benchmark AR(1) coefficient
MAX_LAGS = 4          # cap on BIC lag search for the AR-GAP order


_ANCHOR_CACHE: dict[str, pd.Series | None] = {}


def _expinf10() -> pd.Series | None:
    """Fetch Cleveland Fed 10-yr expected inflation, cached across calls."""
    if "expinf10" in _ANCHOR_CACHE:
        return _ANCHOR_CACHE["expinf10"]
    try:
        from ..data import fred as _f
        s = _f._fetch_fred_series("EXPINF10YR", "1960-01-01")
    except Exception:
        s = None
    _ANCHOR_CACHE["expinf10"] = s
    return s


def _spf_cpi10() -> pd.Series | None:
    """SPF 10-year-ahead CPI forecast (Philly Fed), cached across calls.

    Primary Blue-Chip replacement for τ_t — same object (long-horizon
    professional-consensus inflation forecast), quarterly, 1991Q4→.
    """
    if "spf10" in _ANCHOR_CACHE:
        return _ANCHOR_CACHE["spf10"]
    try:
        from ..data.surveys import load_spf_cpi10
        s = load_spf_cpi10()
    except Exception:
        s = None
    _ANCHOR_CACHE["spf10"] = s
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
    """Local-mean trend τ_t with ordered fallbacks: SPF CPI10 → EXPINF10YR → exp-smoothing.

    Returns a Series aligned to `pi`. Priority: SPF 10-yr CPI (Blue Chip stand-in)
    wherever available, Cleveland Fed EXPINF10YR to fill earlier post-1982 gaps,
    and FW's α=0.95 exp-smoothing seed for the earliest pre-anchor sample.
    """
    pi = pi.astype(float).dropna()
    if len(pi) == 0:
        return pd.Series(dtype=float, name="tau")

    # Detect pi's cadence for resampling anchor series.
    freq = (getattr(pi.index, "freqstr", None) or "").upper()
    if not freq and len(pi) > 1:
        median_days = float(np.median(np.diff(pi.index.asi8)) / 86_400e9)
        freq = "Q" if median_days > 45 else "M"
    rule = "QS" if freq.startswith("Q") else "MS"

    def _align(s: pd.Series | None) -> pd.Series | None:
        if s is None or s.empty:
            return None
        return s.resample(rule).mean().reindex(pi.index).ffill()

    spf10 = _align(_spf_cpi10())
    exp10 = _align(_expinf10())
    tau_es = _exp_smooth(pi, alpha)

    # Layer in priority order: exp-smoothing base, then EXPINF10YR where available,
    # then SPF CPI10 (highest priority) where available.
    tau = tau_es.copy()
    if exp10 is not None:
        mask = exp10.notna()
        tau.loc[mask] = exp10.loc[mask]
    if spf10 is not None:
        mask = spf10.notna()
        tau.loc[mask] = spf10.loc[mask]
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
