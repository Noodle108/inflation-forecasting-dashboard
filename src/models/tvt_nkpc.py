"""New Keynesian Phillips Curve with a time-varying trend (TVT-NKPC).

Standard Phillips-curve forecasts assume inflation reverts to a *fixed* mean, which is
why they failed when trend inflation shifted (the 1970s, and again after 2020). Following
Cogley–Sbordone (2008) and the time-varying-trend NKPC literature, this model separates
inflation into a slow-moving **trend** and a stationary **gap** that obeys a Phillips
curve:

    pi_t          = tau_t + gap_t
    gap_t         = rho * gap_{t-1} - lambda * ugap_t + eps_t   (NKPC gap equation)

**Anchor choice**: the forecast is the *future* trend + the mean-reverting gap. We use
the **Cleveland Fed 10-year expected inflation** (EXPINF10YR) as the trend when
available — that's the market/survey anchor Bernanke–Blanchard (2023) use in their
wage–price system. If EXPINF10YR isn't in the activity frame we fall back to an
exponentially smoothed local-mean (Faust–Wright α=0.95 style), which shrinks toward the
long-run inflation mean rather than snapping to today's print.

This fixes a bug where the previous local-level Kalman filter identified τ_T ≈ π_T
because the state innovation variance was fit large — the model would forecast
inflation to stay at wherever it happens to be *today*, which is not what a
"time-varying-trend" specification is supposed to do.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo


def _exp_smooth(x: pd.Series, alpha: float = 0.95) -> pd.Series:
    """τ_t = α τ_{t-1} + (1-α) x_t. Matches Faust–Wright footnote 8."""
    x = x.astype(float).dropna()
    out = np.empty(len(x))
    if len(x) == 0:
        return pd.Series(out, index=x.index)
    out[0] = float(x.iloc[0])
    for t in range(1, len(x)):
        out[t] = alpha * out[t - 1] + (1 - alpha) * float(x.iloc[t])
    return pd.Series(out, index=x.index, name="tau")


class TVTNKPC(ForecastModel):
    info = ModelInfo(
        key="tvtnkpc",
        name="Phillips curve, time-varying trend (TVT-NKPC)",
        family="Structural",
        reference="Cogley–Sbordone (2008); Bernanke–Blanchard (2023) anchor",
        description=(
            "A New Keynesian Phillips curve written around a slowly evolving anchor: "
            "inflation = trend τ_t + stationary gap driven by economic slack. The trend "
            "is the Cleveland Fed 10-year expected-inflation series (EXPINF10YR) when "
            "available — i.e. long-run market/survey expectations — with an "
            "exponentially-smoothed local-mean fallback. Because the anchor is τ, "
            "long-horizon forecasts converge to expected inflation (~2.5% currently), "
            "not to the last inflation print."
        ),
        needs_activity=True,
        citation="Cogley, T. & Sbordone, A. (2008), 'Trend Inflation, Indexation, and Inflation Persistence in the New Keynesian Phillips Curve', American Economic Review 98(5).",
        intuition="Splits inflation into 'where expectations are anchored' (a slow-moving τ) and 'where the cycle pushes it' (a slack-driven gap). Forecasts both parts and adds them.",
        unique="The only Phillips-curve variant here whose long-run anchor is a survey/market-implied expected inflation rather than the sample mean.",
        strengths="Robust to shifts in trend inflation; long-run forecast tracks expectations by construction.",
        caveats="If EXPINF10YR isn't available (pre-1982) the anchor falls back to exponentially smoothed inflation, which can drift with recent prints during high-inflation episodes.",
        forecast_shape="Glides from the current inflation gap to the anchor τ_T as slack normalizes.",
    )

    def __init__(self, activity_col: str = "ngap", anchor_col: str = "exp10yr",
                 alpha_fallback: float = 0.95):
        super().__init__(activity_col=activity_col, anchor_col=anchor_col,
                         alpha_fallback=alpha_fallback)
        self.activity_col = activity_col
        self.anchor_col = anchor_col
        self.alpha_fallback = alpha_fallback

    def _fit(self) -> None:
        import statsmodels.api as sm

        y = self._y
        Xdf = self._X if self._X is not None else pd.DataFrame(index=y.index)

        # ---- Trend: prefer EXPINF10YR; fall back to exponentially smoothed y. ----
        if self.anchor_col in getattr(Xdf, "columns", []):
            tau_raw = Xdf[self.anchor_col].reindex(y.index)
            # ffill forward; bfill only what's needed at the far left of the sample
            tau = tau_raw.ffill()
            if tau.isna().any():
                tau_es = _exp_smooth(y, self.alpha_fallback)
                tau = tau.combine_first(tau_es)
            self._anchor_source = "EXPINF10YR"
        else:
            tau = _exp_smooth(y, self.alpha_fallback)
            self._anchor_source = f"exp-smooth(α={self.alpha_fallback})"

        self._trend = float(tau.iloc[-1])
        gap = (y - tau).rename("gap")

        if self.activity_col in getattr(Xdf, "columns", []):
            ugap = Xdf[self.activity_col].reindex(y.index).ffill()
        else:
            ugap = pd.Series(0.0, index=y.index)

        d = pd.DataFrame({"gap": gap, "ugap": ugap})
        d["gap_l1"] = d["gap"].shift(1)
        dd = d.dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xg = sm.add_constant(dd[["gap_l1", "ugap"]].values)
            self._gap_eq = sm.OLS(dd["gap"].values, Xg).fit()
            a = pd.DataFrame({"u": ugap}); a["u_l1"] = a["u"].shift(1)
            ad = a.dropna()
            self._ugap_ar = sm.OLS(ad["u"].values,
                                    sm.add_constant(ad["u_l1"].values)).fit()

        self._last_gap = float(gap.iloc[-1])
        self._last_ugap = float(ugap.iloc[-1])

    def _forecast(self, h: int) -> float:
        c_g, rho, lam = self._gap_eq.params
        c_u, b_u = self._ugap_ar.params
        gap, ugap = self._last_gap, self._last_ugap
        for _ in range(h):
            ugap = c_u + b_u * ugap
            gap = c_g + rho * gap + lam * ugap
        return self._trend + gap

    def steady_state(self) -> float:
        """Anchor plus the gap's implied unconditional mean."""
        c_g, rho, lam = self._gap_eq.params
        c_u, b_u = self._ugap_ar.params
        if abs(rho) >= 1 or abs(b_u) >= 1:
            return float(self._trend)
        ugap_ss = c_u / (1 - b_u)
        gap_ss = (c_g + lam * ugap_ss) / (1 - rho)
        return float(self._trend + gap_ss)
