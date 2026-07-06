"""Cleveland Fed inflation nowcasting model.

The Cleveland Fed publishes monthly *inflation nowcasts* for CPI and PCE, built by a
small bridge model that exploits **high-frequency indicators available before the
official price release**. The dominant such indicator is weekly retail gasoline prices
(GASREGW), which land on Monday and lead the monthly CPI release by ~3 weeks. Because
energy is roughly 7% of the CPI basket, gasoline alone explains a large share of
month-to-month headline surprises; adding lags of realized inflation and a food-price
proxy soaks up the rest.

Following the Knotek–Zaman (2017) specification, this is a **direct bridge regression**
of monthly inflation on: (i) the current-month growth in retail gasoline (properly
aggregated from weekly to monthly and log-differenced), (ii) lagged inflation, and
(iii) a food-price growth term. At forecast time gasoline growth is either observed
(if the target month has begun) or projected forward as an AR(1) — the same bridge
mechanic the Cleveland Fed uses when the month is only partly complete.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo


def _load_nowcast_data(freq: str = "M"):
    """Assemble the monthly (or quarterly) bridge dataset from FRED."""
    from ..data import fred as _f
    if not _f.has_live_data():
        raise RuntimeError("Cleveland Fed nowcast needs live FRED data.")

    start = "1995-01-01"
    cpi = _f._fetch_fred_series("CPIAUCSL", start)          # monthly level
    gas = _f._fetch_fred_series("GASREGW", start)           # weekly level (Monday obs)
    food = _f._fetch_fred_series("CPIUFDSL", start)         # monthly level

    if cpi is None or gas is None or food is None:
        raise RuntimeError("Cleveland Fed nowcast: missing CPIAUCSL / GASREGW / CPIUFDSL.")

    if freq == "Q":
        cpi = cpi.resample("QS").mean()
        food = food.resample("QS").mean()
        gas_m = gas.resample("QS").mean()
    else:
        gas_m = gas.resample("MS").mean()                    # weekly → monthly mean

    factor = 1200.0 if freq == "M" else 400.0
    pi = factor * np.log(cpi / cpi.shift(1))
    g_gas = factor * np.log(gas_m / gas_m.shift(1))
    g_food = factor * np.log(food / food.shift(1))

    df = pd.DataFrame({"pi": pi, "g_gas": g_gas, "g_food": g_food}).dropna()
    return df


class ClevelandNowcast(ForecastModel):
    info = ModelInfo(
        key="clenow",
        name="Cleveland Fed inflation nowcast",
        family="Statistical",
        reference="Knotek–Zaman (2017); Cleveland Fed nowcast",
        description=(
            "A bridge regression of headline CPI inflation on the current month's growth "
            "in retail gasoline prices (weekly, available in real time), the previous "
            "month's inflation, and food-price growth. Reproduces the mechanic behind "
            "the Cleveland Fed's monthly inflation nowcast: gasoline lands weeks before "
            "the CPI release, so aggregating those weekly prints into a monthly average "
            "yields a sharp same-month forecast; farther-out forecasts iterate gasoline "
            "growth forward as an AR(1)."
        ),
        needs_activity=False,
        citation="Knotek II, E. & Zaman, S. (2017), 'Nowcasting US Headline and Core Inflation', Journal of Money, Credit and Banking 49(5); federalreserve.cleveland.org/indicators-and-data/inflation-nowcasting.",
        intuition="Retail gasoline prints land ~3 weeks before the CPI release. Aggregate them into a monthly average and you already know most of what the energy line will contribute to the CPI print.",
        unique="The only model here designed for the *very short* horizon — the current or next month's inflation print — using real-time high-frequency data (weekly gasoline) rather than only monthly variables.",
        strengths="At h=1 (the nowcast horizon it was built for) it typically beats random-walk and AR benchmarks by a wide margin because it exploits information a monthly model cannot see.",
        caveats="Advantage decays fast beyond one or two months as the gasoline signal must be projected forward; not designed as a medium-horizon forecast.",
        forecast_shape="Sharp jump toward the gasoline-implied print in the first period, then gliding back toward the average as gasoline growth mean-reverts.",
    )

    def __init__(self, n_lags: int = 2, freq: str = "M"):
        super().__init__(n_lags=n_lags, freq=freq)
        self.n_lags = n_lags
        self.freq = freq

    def _fit(self) -> None:
        import statsmodels.api as sm

        # Determine target frequency from the fitted pi series.
        pi_index = self._y.index
        is_quarterly = bool(pi_index.freqstr and pi_index.freqstr.startswith("Q"))
        freq = "Q" if is_quarterly else "M"

        df = _load_nowcast_data(freq=freq)
        # Align to the training window so backtest at earlier origins is honest.
        df = df.loc[:pi_index[-1]]
        self._df = df

        p = self.n_lags
        d = pd.DataFrame(index=df.index)
        d["y"] = df["pi"]
        d["g_gas"] = df["g_gas"]
        d["g_food"] = df["g_food"]
        for i in range(1, p + 1):
            d[f"pi_l{i}"] = df["pi"].shift(i)
            d[f"gas_l{i}"] = df["g_gas"].shift(i)
        dd = d.dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xp = sm.add_constant(dd.drop(columns="y").values)
            self._bridge = sm.OLS(dd["y"].values, Xp).fit()

            # AR(1) for gasoline growth (to project it forward beyond h=1)
            g = pd.DataFrame({"g": df["g_gas"]}); g["l1"] = g["g"].shift(1)
            gd = g.dropna()
            self._gas_ar = sm.OLS(gd["g"].values,
                                  sm.add_constant(gd["l1"].values)).fit()
            # AR(1) for food growth as well
            f = pd.DataFrame({"g": df["g_food"]}); f["l1"] = f["g"].shift(1)
            fd = f.dropna()
            self._food_ar = sm.OLS(fd["g"].values,
                                   sm.add_constant(fd["l1"].values)).fit()

        self._pi_hist = list(df["pi"].values[-p:])
        self._gas_hist = list(df["g_gas"].values[-p:])
        self._last_gas = float(df["g_gas"].iloc[-1])
        self._last_food = float(df["g_food"].iloc[-1])

    def _forecast(self, h: int) -> float:
        c_g, rho_g = self._gas_ar.params
        c_f, rho_f = self._food_ar.params
        pc = self._bridge.params
        p = self.n_lags
        pi_hist = list(self._pi_hist)
        gas_hist = list(self._gas_hist)
        g_food = self._last_food
        g_gas_next = self._last_gas   # first step already-observed component
        pi_new = pi_hist[-1]
        for step in range(h):
            g_gas_next = c_g + rho_g * g_gas_next if step > 0 else g_gas_next
            g_food = c_f + rho_f * g_food
            # bridge regression: [const, g_gas, g_food, pi_l1..p, gas_l1..p]
            x = [1.0, g_gas_next, g_food]
            for i in range(1, p + 1):
                x.append(pi_hist[-i])
            for i in range(1, p + 1):
                x.append(gas_hist[-i])
            pi_new = float(np.dot(pc, x))
            pi_hist.append(pi_new)
            gas_hist.append(g_gas_next)
        return pi_new
