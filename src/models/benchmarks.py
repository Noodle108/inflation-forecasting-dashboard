"""Benchmark models.

The random walk and the Atkeson–Ohanian (2001) rolling mean are the yardsticks the
whole literature is measured against — famously hard to beat for US inflation.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastModel, ModelInfo


class RandomWalk(ForecastModel):
    """`pi_{t+h|t} = pi_t`. The naive no-change forecast."""

    info = ModelInfo(
        key="rw",
        name="Random Walk (no change)",
        family="Benchmark",
        reference="Atkeson–Ohanian (2001)",
        description=(
            "Forecasts future inflation as equal to the most recent observation. "
            "The simplest possible benchmark; under an IMA(1,1) / UCSV data-generating "
            "process it is close to optimal, which is why it is so hard to beat."
        ),
        citation="Atkeson, A. & Ohanian, L. (2001), 'Are Phillips Curves Useful for Forecasting Inflation?', Minneapolis Fed QR.",
        intuition="Takes today's inflation rate and carries it forward unchanged for every future period.",
        unique="The only model here with zero parameters and no estimation — pure 'no-change'. Every other model is judged against it.",
        strengths="Very hard to beat since the mid-1980s, because inflation behaves close to a random walk with a slowly drifting trend.",
        caveats="Ignores mean-reversion and any information from the economy; whipsaws on one-off spikes in a single month's print.",
        forecast_shape="A flat horizontal line at the last observed value.",
    )

    def _fit(self) -> None:
        self._last = float(self._y.iloc[-1])

    def _forecast(self, h: int) -> float:
        return self._last


class AtkesonOhanian(ForecastModel):
    """`pi_{t+h|t} = mean(pi over last `window` periods)`.

    The Atkeson–Ohanian benchmark: last four quarters (or 12 months) of inflation.
    """

    info = ModelInfo(
        key="ao",
        name="Atkeson–Ohanian (rolling mean)",
        family="Benchmark",
        reference="Atkeson–Ohanian (2001)",
        description=(
            "Forecasts inflation over the next year as the average inflation rate over "
            "the previous four quarters (12 months). Atkeson & Ohanian showed this "
            "simple average beats estimated Phillips-curve forecasts since ~1985."
        ),
        citation="Atkeson, A. & Ohanian, L. (2001), 'Are Phillips Curves Useful for Forecasting Inflation?', Minneapolis Fed QR.",
        intuition="Averages the last year of inflation and projects that average forward — a smoothed version of the random walk.",
        unique="Like the random walk but it averages away one-month noise first, so it reacts less to a single volatile print.",
        strengths="The headline result of the paper: this trivial average out-forecasts estimated Phillips curves over 1985–2000. A stern benchmark.",
        caveats="Still purely backward-looking; lags turning points because the 12-month window is slow to update.",
        forecast_shape="A flat horizontal line at the trailing 12-month average.",
    )

    def __init__(self, window: int | None = None):
        # None → infer 4 (quarterly) / 12 (monthly) from the series at fit time.
        super().__init__(window=window)
        self.window = window

    def _fit(self) -> None:
        w = self.window
        if w is None:
            # Prefer the pandas-attached freq; fall back to the median date-diff,
            # since dropna() upstream can wipe freqstr even on regularly-spaced data.
            freq = (getattr(self._y.index, "freqstr", None) or "").upper()
            if not freq and len(self._y) > 1:
                median_days = float(np.median(np.diff(self._y.index.asi8)) / 86_400e9)
                freq = "Q" if median_days > 45 else "M"
            w = 4 if freq.startswith("Q") else 12
        w = min(w, len(self._y))
        self._mean = float(self._y.iloc[-w:].mean())

    def _forecast(self, h: int) -> float:
        return self._mean
