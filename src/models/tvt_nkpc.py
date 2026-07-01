"""New Keynesian Phillips Curve with a time-varying trend (TVT-NKPC).

Standard Phillips-curve forecasts assume inflation reverts to a *fixed* mean, which is
why they failed when trend inflation shifted (the 1970s, and again after 2020). Following
Cogley–Sbordone (2008) and the time-varying-trend NKPC literature, this model separates
inflation into a slow-moving stochastic **trend** and a stationary **gap** that obeys a
Phillips curve:

    pi_t          = tau_t + gap_t
    tau_t         = tau_{t-1} + eta_t                 (random-walk trend)
    gap_t         = rho * gap_{t-1} - lambda * ugap_t + eps_t   (NKPC gap equation)

Inflation is forecast as the current trend plus the mean-reverting gap driven by economic
slack. Because the long-run anchor is the *drifting* trend rather than a constant, the
model tracks shifts in underlying inflation that break fixed-mean Phillips curves.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo


class TVTNKPC(ForecastModel):
    info = ModelInfo(
        key="tvtnkpc",
        name="Phillips curve, time-varying trend (TVT-NKPC)",
        family="Structural",
        reference="Cogley–Sbordone (2008)",
        description=(
            "A New Keynesian Phillips curve written around a drifting stochastic trend: "
            "inflation equals a slow-moving trend plus a stationary gap that responds to "
            "economic slack (the unemployment gap). The forecast is the current trend "
            "plus the mean-reverting gap. Fixes the fixed-mean assumption that makes "
            "ordinary Phillips curves miss shifts in underlying inflation."
        ),
        needs_activity=True,
        citation="Cogley, T. & Sbordone, A. (2008), 'Trend Inflation, Indexation, and Inflation Persistence in the New Keynesian Phillips Curve', American Economic Review 98(5).",
        intuition="Splits inflation into 'where it's anchored' (a drifting trend) and 'where the cycle pushes it' (a slack-driven gap), and forecasts both parts.",
        unique="Combines the UCSV trend with a Phillips curve: unlike the plain Phillips curve it does not revert to a fixed mean, and unlike UCSV it lets economic slack move the near-term forecast.",
        strengths="Robust to shifts in trend inflation (1970s, post-2020) that break fixed-mean Phillips curves, while still using activity information.",
        caveats="Trend and gap are estimated sequentially (trend via a Kalman local level, gap via OLS) rather than jointly; the slack–inflation link is still weak in recent data.",
        forecast_shape="Starts from the current inflation gap and glides to the estimated trend as slack normalizes.",
    )

    def __init__(self, activity_col: str = "ngap"):
        super().__init__(activity_col=activity_col)
        self.activity_col = activity_col

    def _fit(self) -> None:
        import statsmodels.api as sm
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        y = self._y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            uc = UnobservedComponents(y.reset_index(drop=True), level="local level").fit(disp=False)
        tau = pd.Series(np.asarray(uc.smoothed_state[0]), index=y.index)
        self._trend = float(tau.iloc[-1])
        gap = (y - tau).rename("gap")

        has_act = self._X is not None and self.activity_col in getattr(self._X, "columns", [])
        if has_act:
            ugap = self._X[self.activity_col].reindex(y.index).ffill()
        else:
            ugap = pd.Series(0.0, index=y.index)

        d = pd.DataFrame({"gap": gap, "ugap": ugap})
        d["gap_l1"] = d["gap"].shift(1)
        dd = d.dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xg = sm.add_constant(dd[["gap_l1", "ugap"]].values)
            self._gap_eq = sm.OLS(dd["gap"].values, Xg).fit()
            # AR(1) for the activity gap so it can be projected forward
            a = pd.DataFrame({"u": ugap}); a["u_l1"] = a["u"].shift(1)
            ad = a.dropna()
            self._ugap_ar = sm.OLS(ad["u"].values, sm.add_constant(ad["u_l1"].values)).fit()

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
