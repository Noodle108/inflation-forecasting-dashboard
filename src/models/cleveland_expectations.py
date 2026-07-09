"""Cleveland Fed inflation-expectations model.

The Cleveland Fed publishes model-based measures of expected inflation at horizons
1, 2, 3, 5, 7, 10, 20, and 30 years, produced by the term-structure model of Haubrich,
Pennacchi & Ritchken (2012). The model jointly fits nominal Treasury yields, TIPS
yields, inflation-swap rates, and survey expectations, extracting a **market-implied
term structure of expected inflation** with the inflation risk premium stripped out.

For forecasting inflation at horizon h, the natural forecast is the average inflation
rate expected over the next h periods, which is exactly what the ``EXPINF*YR`` FRED
series provides — the Cleveland Fed does the estimation for us and publishes it in
real time. This "model" is therefore a pass-through wrapper: at fit time it caches the
latest term structure, and at forecast time it maps horizon h to the appropriate
maturity by log-linear interpolation. It is the *market's* forecast, delivered to the
dashboard on a common footing with the statistical and structural models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo

# FRED series ids: expected-inflation term structure (annualized %, monthly).
_EI_TERMS = [
    (1, "EXPINF1YR"), (2, "EXPINF2YR"), (3, "EXPINF3YR"),
    (5, "EXPINF5YR"), (7, "EXPINF7YR"), (10, "EXPINF10YR"),
    (20, "EXPINF20YR"), (30, "EXPINF30YR"),
]


def _load_expectations(as_of=None) -> pd.DataFrame:
    """Return a DataFrame indexed by month, columns = maturities in years."""
    from ..data import fred as _f
    if not _f.has_live_data():
        raise RuntimeError("Cleveland Fed expectations need live FRED data.")
    frames = {}
    for tau, sid in _EI_TERMS:
        s = _f._fetch_fred_series(sid, "1982-01-01")
        if s is not None:
            frames[tau] = s
    if not frames:
        raise RuntimeError("Cleveland Fed expectations: no EXPINF*YR series returned.")
    df = pd.DataFrame(frames).sort_index()
    if as_of is not None:
        df = df.loc[:as_of]
    return df.dropna(how="all")


class ClevelandExpectations(ForecastModel):
    info = ModelInfo(
        key="cleexp",
        name="Cleveland Fed inflation expectations",
        family="Structural",
        reference="Haubrich–Pennacchi–Ritchken (2012); Cleveland Fed",
        description=(
            "The market's own inflation forecast, delivered by the Cleveland Fed's "
            "term-structure model. Haubrich, Pennacchi & Ritchken jointly fit nominal "
            "Treasury yields, TIPS yields, inflation swaps, and survey expectations to "
            "extract expected inflation at horizons 1–30 years (net of the inflation "
            "risk premium). Forecasts at horizon h return the expected average inflation "
            "over that horizon by interpolating the term structure."
        ),
        needs_activity=False,
        citation="Haubrich, J., Pennacchi, G. & Ritchken, P. (2012), 'Inflation Expectations, Real Rates, and Risk Premia', Review of Financial Studies 25(5); clevelandfed.org/indicators-and-data/inflation-expectations.",
        intuition="Reads inflation expectations directly out of asset prices — what the bond and swap markets are collectively pricing for inflation over the next h years, purged of the risk premium.",
        unique="The only model here whose forecast comes from *financial-market prices* rather than from macro time-series or a structural model — a forward-looking anchor for the other models.",
        strengths="Real-time and forward-looking; the term-structure fit means the h=1 vs h=10 forecast come from the same coherent model rather than being unrelated numbers.",
        caveats="Fits an average over the horizon, not a point-in-time value; risk-premium identification depends on the joint use of TIPS and surveys, so tails can look overly smooth.",
        forecast_shape="Follows the market-implied inflation curve — typically flatter than model-based paths, gliding smoothly from 1-year to long-run expectations.",
    )

    def __init__(self):
        super().__init__()
        self._curve: pd.DataFrame | None = None

    def _fit(self) -> None:
        # Cache the expectations term structure up to (and including) the last
        # observation date in the training set — so backtest origins are honest.
        as_of = self._y.index[-1]
        curve = _load_expectations(as_of=as_of)
        if curve.empty:
            raise RuntimeError("Cleveland expectations: empty term structure at as-of date.")
        self._curve = curve
        self._latest = curve.iloc[-1].dropna()
        # Latest realized inflation — used as the h=0 anchor when interpolating
        # sub-1yr horizons (the Cleveland Fed's shortest series is 1yr).
        self._last_realized = float(self._y.iloc[-1])
        # Determine periods-per-year from the training frequency
        is_quarterly = bool(self._y.index.freqstr and self._y.index.freqstr.startswith("Q"))
        self._ppy = 4 if is_quarterly else 12

    def _forecast(self, h: int) -> float:
        # h is in *periods*; convert to years (fractional allowed).
        tau_h = max(h / self._ppy, 1e-3)
        taus = np.asarray(self._latest.index, dtype=float)
        vals = self._latest.values.astype(float)
        order = np.argsort(taus)
        taus, vals = taus[order], vals[order]
        # Below 1yr: the Cleveland Fed doesn't publish a shorter maturity, so
        # linearly interpolate between the *current realized print* (h=0 anchor)
        # and the 1-year expected value. This turns the flat sub-1yr region into
        # a visible glide from where inflation is now toward the 1yr expectation.
        if tau_h < taus[0]:
            w = tau_h / taus[0]                       # 0 at h=0, 1 at 1yr
            return float((1 - w) * self._last_realized + w * vals[0])
        # Above the longest published maturity: clamp (avoids extrapolation).
        if tau_h > taus[-1]:
            return float(vals[-1])
        return float(np.interp(tau_h, taus, vals))

    def steady_state(self) -> float:
        """Long-run inflation anchor = the longest-maturity expectation available."""
        vals = self._latest.values.astype(float)
        taus = np.asarray(self._latest.index, dtype=float)
        return float(vals[np.argmax(taus)])
