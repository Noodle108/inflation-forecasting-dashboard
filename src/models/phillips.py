"""Phillips-curve forecast.

Backward-looking (accelerationist) Phillips curve in the Stock–Watson tradition:
inflation is regressed on its own lags plus lags of an activity/slack variable
(here the unemployment gap). Estimated as a *direct* h-step forecasting regression.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo


class PhillipsCurve(ForecastModel):
    info = ModelInfo(
        key="pc",
        name="Phillips Curve (activity-based)",
        family="Statistical",
        reference="Stock–Watson (1999, 2008)",
        description=(
            "Regresses inflation on its own lags and lags of economic slack (the "
            "unemployment gap). The classic activity-based forecast. Direct h-step "
            "estimation. Historically strong in the 1970s–80s but often beaten by the "
            "random walk since the mid-1980s — a key finding this dashboard lets you check."
        ),
        needs_activity=True,
        citation="Stock, J. & Watson, M. (1999, JME; 2008, NBER wp) — activity-based Phillips-curve forecasts.",
        intuition="Says inflation pressure rises when the economy runs hot (low unemployment gap) and eases when it runs cold, on top of inflation's own momentum.",
        unique="The only reduced-form model here that brings in the real economy (labor-market slack) rather than inflation's history alone.",
        strengths="Adds genuine information at turning points when slack is large — historically strong in the 1970s–80s.",
        caveats="The slack–inflation link flattened after ~1985; Atkeson–Ohanian showed it then fails to beat a simple average out of sample.",
        forecast_shape="Tracks the AR path but tilts up or down depending on the current unemployment gap.",
    )

    def __init__(self, n_lags: int = 4, activity_col: str = "ngap"):
        super().__init__(n_lags=n_lags, activity_col=activity_col)
        self.n_lags = n_lags
        self.activity_col = activity_col

    def _fit(self) -> None:
        # Requires an activity regressor; fall back to a pure AR if missing.
        self._has_activity = (
            self._X is not None and self.activity_col in getattr(self._X, "columns", [])
        )
        # Build the design matrix lazily at forecast time per horizon (direct method),
        # so just stash aligned series here.
        self._pi = self._y
        if self._has_activity:
            self._act = self._X[self.activity_col].reindex(self._pi.index).ffill()
        else:
            self._act = None

    def _design(self, h: int):
        """Direct h-step design: y_{t} on pi_{t-h..t-h-p+1} and act_{t-h..}."""
        import statsmodels.api as sm

        p = self.n_lags
        df = pd.DataFrame({"pi": self._pi.values}, index=self._pi.index)
        cols = {}
        for l in range(p):
            cols[f"pi_l{l}"] = df["pi"].shift(h + l)
        if self._act is not None:
            a = self._act.reindex(df.index).ffill()
            for l in range(p):
                cols[f"a_l{l}"] = a.shift(h + l)
        Xd = pd.DataFrame(cols)
        data = pd.concat([df["pi"], Xd], axis=1).dropna()
        y = data["pi"].values
        X = sm.add_constant(data.drop(columns="pi").values)
        return y, X

    def _forecast(self, h: int) -> float:
        import statsmodels.api as sm

        y, X = self._design(h)
        if len(y) < X.shape[1] + 5:  # too little data → fall back to last value
            return float(self._pi.iloc[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(y, X).fit()

        # Build the regressor row from the *most recent* available data.
        p = self.n_lags
        row = [1.0]
        for l in range(p):
            row.append(float(self._pi.iloc[-1 - l]))
        if self._act is not None:
            for l in range(p):
                row.append(float(self._act.iloc[-1 - l]))
        row = np.asarray(row).reshape(1, -1)
        return float(res.predict(row)[0])
