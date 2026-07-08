"""Shared utilities for the Faust–Wright (2013) forecasting suite.

Faust & Wright's central device is the **local-mean trend** τ_t, around which each
forecast is written in "gap" form (g_t = π_t − τ_t). Their preferred τ_t is the most
recent Blue-Chip 5–10y long-run inflation forecast; pre-1979, they fall back to
exponential smoothing (α = 0.95) of realized inflation. Blue-Chip data is not on FRED,
so we use FW's own exponential-smoothing definition throughout this build — the same
formula they document in footnote 8:

    τ_t = α τ_{t-1} + (1 − α) π_t,     α = 0.95.

Also collected here: BIC lag selection for AR-GAP, and a tiny direct h-step OLS helper
that most FW models share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ALPHA_TAU = 0.95      # FW's exponential-smoothing coefficient (footnote 8)
FIXED_RHO = 0.46      # FW's fixed-ρ benchmark AR(1) coefficient
MAX_LAGS = 4          # cap on BIC lag search for the AR-GAP order


def local_mean_trend(pi: pd.Series, alpha: float = ALPHA_TAU) -> pd.Series:
    """Exponentially smoothed local mean of inflation. Returns a Series aligned to pi.

    Uses FW's convention τ_t = α τ_{t-1} + (1 − α) π_t (footnote 8). The recursion is
    seeded with the first observation.
    """
    pi = pi.astype(float).dropna()
    out = np.empty(len(pi))
    out[0] = float(pi.iloc[0])
    for t in range(1, len(pi)):
        out[t] = alpha * out[t - 1] + (1.0 - alpha) * float(pi.iloc[t])
    return pd.Series(out, index=pi.index, name="tau")


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
