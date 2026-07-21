"""Shared utilities for the Faust–Wright (2013) forecasting suite.

Faust & Wright's central device is the **local-mean trend** τ_t, around which each
forecast is written in "gap" form (g_t = π_t − τ_t). Their preferred τ_t is the most
recent Blue-Chip 5–10y long-run inflation forecast (available from 1979); pre-1979
they fall back to exponential smoothing (α = 0.95) of realized inflation (footnote 8).

We build τ_t from four ordered fallbacks (highest priority first, measure-aware):

    1. **Blue Chip 5–10y** for the *matching measure* (Table 1.2 uses GDPDEF;
       Table 1.3 uses CPI). Read from data/surveys/blue_chip_lr.xlsx if present.
       This is the paper-exact anchor when the file is available.
    2. **SPF 10-year CPI** (Philly Fed sheet ``CPI10``, 1991Q4→). Open-source
       Blue-Chip stand-in — quarterly professional consensus at a 5–10y horizon.
       Used only when the measure is CPI (mixing CPI anchors into GDPDEF/PCE
       forecasts introduces a persistent 30–50 bp wedge, so we don't).
    3. **Cleveland Fed 10-yr expected inflation** (FRED ``EXPINF10YR``, 1982→).
       Model-based (Haubrich–Pennacchi–Ritchken 2012), also CPI-flavored — used
       to bridge SPF's 1991Q4 start when the measure is CPI.
    4. **FW's exponential-smoothing fallback** (footnote 8):
           τ_t = α τ_{t-1} + (1 − α) π_t,     α = 0.95
       Universal fallback — used before survey anchors exist, and used for
       GDPDEF/PCE whenever Blue Chip LR is not available.

All anchors are annualized percent — same units as π_t, no scaling needed.

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

# Which π-series names route to which Blue-Chip LR measure. Missing keys fall
# through to no BC-LR anchor (e.g. "core_cpi", "core_pce" — the paper's Table 1.2
# / 1.3 don't cover core aggregates).
_BC_MEASURE_FOR = {
    "cpi":    "cpi",
    "pce":    "pce",
    "gdpdef": "gdpdef",
    "pgdp":   "gdpdef",
}
# Series names for which CPI-flavored open-source anchors (SPF CPI10, EXPINF10YR)
# are a reasonable stand-in when Blue Chip LR is missing.
_CPI_FLAVORED = {"cpi", "core_cpi"}


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

    Open-source Blue-Chip replacement for τ_t when the inflation measure being
    forecast is CPI-flavored — same object (long-horizon professional-consensus
    inflation forecast), quarterly, 1991Q4→.
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


def _bc_lr(measure: str) -> pd.Series | None:
    """Blue Chip 5-10y long-range forecast for a specific measure, cached."""
    ck = f"bc_lr_{measure}"
    if ck in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[ck]
    try:
        from ..data.surveys import load_blue_chip_lr
        s = load_blue_chip_lr(measure)
    except Exception:
        s = None
    _ANCHOR_CACHE[ck] = s
    return s


def _exp_smooth(pi: pd.Series, alpha: float) -> pd.Series:
    """τ_t = α τ_{t-1} + (1-α) π_t, seeded at π_0. FW footnote 8."""
    vals = pi.values.astype(float)
    out = np.empty(len(vals))
    out[0] = vals[0]
    for t in range(1, len(vals)):
        out[t] = alpha * out[t - 1] + (1.0 - alpha) * vals[t]
    return pd.Series(out, index=pi.index, name="tau")


def local_mean_trend(pi: pd.Series, alpha: float = ALPHA_TAU,
                     measure: str | None = None) -> pd.Series:
    """Measure-aware local-mean trend τ_t.

    Priority (highest first):
      1. Blue Chip 5-10y for the matching measure (paper-exact, if the file
         data/surveys/blue_chip_lr.xlsx is present).
      2. SPF CPI10, then EXPINF10YR — only when the measure is CPI-flavored.
         Mixing CPI anchors into GDPDEF/PCE gap forecasts biases the trend by
         30-50 bp, so we don't.
      3. FW's α=0.95 exp-smoothing fallback (footnote 8). Universal.

    ``measure`` is auto-detected from ``pi.name`` when omitted; pass explicitly
    when the Series has no name (e.g. from verify_fw.py).
    """
    pi = pi.astype(float).dropna()
    if len(pi) == 0:
        return pd.Series(dtype=float, name="tau")

    if measure is None:
        name = (getattr(pi, "name", "") or "").lower()
        measure = name if name else None

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

    # Apply anchors from LOWEST to HIGHEST priority so each layer overwrites
    # the previous where it has coverage.
    tau = _exp_smooth(pi, alpha)                       # universal base (footnote 8)

    if (measure or "") in _CPI_FLAVORED:               # CPI-only open-source anchors
        for anchor in (_expinf10(), _spf_cpi10()):     # EXPINF10YR, then SPF CPI10
            aligned = _align(anchor)
            if aligned is not None:
                tau.loc[aligned.notna()] = aligned[aligned.notna()]

    bc_measure = _BC_MEASURE_FOR.get(measure or "")    # highest priority: BC-LR
    if bc_measure is not None:
        bc = _align(_bc_lr(bc_measure))
        if bc is not None:
            tau.loc[bc.notna()] = bc[bc.notna()]

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
