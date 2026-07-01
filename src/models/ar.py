"""Autoregressive model, AR(p).

A workhorse univariate statistical forecaster. Lag order can be fixed or selected
by BIC (the common Stock–Watson choice, capped at a small max).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo


class ARModel(ForecastModel):
    info = ModelInfo(
        key="ar",
        name="AR(p)",
        family="Statistical",
        reference="Stock–Watson (2007)",
        description=(
            "An autoregression of inflation on its own lags. Lag order p is selected by "
            "BIC up to `max_lags` (or fixed if `p` is given). Iterated multi-step "
            "forecasts. A standard reduced-form benchmark above the naive random walk."
        ),
        citation="Stock, J. & Watson, M. (2007), 'Why Has US Inflation Become Harder to Forecast?', JMCB — AR(p) reference model.",
        intuition="Predicts inflation from a weighted combination of its own recent values; the weights are fit by least squares.",
        unique="The simplest model that captures inflation's own momentum and mean-reversion — no economic variables, just its own past.",
        strengths="Cheap, stable, and captures short-run persistence; a natural step up from the random walk when inflation is mean-reverting.",
        caveats="Assumes a constant mean and fixed dynamics; struggles when the inflation trend itself shifts (1970s, post-2020).",
        forecast_shape="A curve that decays from the last value back toward the estimated long-run mean at a speed set by the AR coefficients.",
    )

    def __init__(self, p: int | None = None, max_lags: int = 12):
        super().__init__(p=p, max_lags=max_lags)
        self.p = p
        self.max_lags = max_lags

    def _fit(self) -> None:
        from statsmodels.tsa.ar_model import AutoReg, ar_select_order

        y = self._y.reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.p is None:
                maxlag = min(self.max_lags, max(1, len(y) // 4))
                sel = ar_select_order(y, maxlag=maxlag, ic="bic", old_names=False)
                self._order = sel.ar_lags if sel.ar_lags else [1]
                self._res = sel.model.fit()
            else:
                self._res = AutoReg(y, lags=self.p, old_names=False).fit()
                self._order = list(range(1, self.p + 1))

    def _forecast(self, h: int) -> float:
        n = len(self._y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._res.predict(start=n, end=n + h - 1)
        return float(np.asarray(fc)[-1])
