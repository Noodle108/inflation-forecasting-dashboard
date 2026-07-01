"""Bernanke–Blanchard (2023) wage–price model.

A semi-structural model of the post-pandemic inflation, following Bernanke & Blanchard,
"What Caused the U.S. Pandemic-Era Inflation?" (2023). Inflation is the outcome of an
interacting **wage–price system** driven by labor-market tightness, supply shocks, and
inflation expectations:

* **Price equation** (mark-up pricing): price inflation depends on its own lags, wage
  inflation (unit-labor-cost pass-through), and the relative prices of energy and food,
  anchored to long-run expected inflation.
* **Wage equation** (wage Phillips curve with catch-up): wage inflation depends on labor-
  market tightness (the vacancy-to-unemployment ratio), a "catch-up" term for past
  realized-vs-expected inflation, and short-run expectations.

Both equations impose the Bernanke–Blanchard long-run restrictions (unit pass-through of
wages to prices; wages and prices anchored to expectations in steady state) by estimating
in expectation-gap form, so the system is dynamically stable.

This is the FRED-implementable version: it uses the vacancy/unemployment ratio (JOLTS),
relative energy and food prices, and survey expectations (Michigan 1-year, Cleveland Fed
10-year). Bernanke–Blanchard's separate *shortages* term (a Google-Trends / GSCPI supply-
chain measure, not on FRED) is omitted — its supply-shock role is partly captured by the
energy and food terms.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo

_BB_FRED = dict(
    cpi="CPIAUCSL", energy="CPIENGSL", food="CPIUFDSL",
    eci="ECIWAG", vac="JTSJOL", unemp="UNEMPLOY",
    exp1="MICH", exp10="EXPINF10YR",
)


def load_bb_data():
    """Quarterly Bernanke–Blanchard dataset from FRED, all in annualized-% units."""
    from ..data import fred as _f

    if not _f.has_live_data():
        raise RuntimeError("Bernanke–Blanchard needs live FRED data (set FRED_API_KEY).")

    start = "1999-01-01"
    raw = {k: _f._fetch_fred_series(sid, start) for k, sid in _BB_FRED.items()}
    missing = [k for k, v in raw.items() if v is None]
    if missing:
        raise RuntimeError(f"FRED fetch failed for: {missing}")

    def q_mean(s):
        return s.resample("QS").mean()

    cpi = q_mean(raw["cpi"]); energy = q_mean(raw["energy"]); food = q_mean(raw["food"])
    eci = raw["eci"].resample("QS").mean()          # already quarterly index
    vac = q_mean(raw["vac"]); unemp = q_mean(raw["unemp"])
    exp1 = q_mean(raw["exp1"]); exp10 = q_mean(raw["exp10"])

    gp = 400 * np.log(cpi).diff()                    # price inflation, annualized %
    gw = 400 * np.log(eci).diff()                    # wage inflation, annualized %
    g_energy = 400 * np.log(energy).diff()
    g_food = 400 * np.log(food).diff()

    df = pd.DataFrame({
        "gp": gp,
        "gw": gw,
        "rpe": g_energy - gp,                         # relative energy-price inflation
        "rpf": g_food - gp,                           # relative food-price inflation
        "vu": np.log(vac / unemp),                    # log vacancy/unemployment ratio
        "Ep": exp1,                                   # short-run expectations
        "Elr": exp10,                                 # long-run expectations
    }).dropna()
    return df


class BernankeBlanchard(ForecastModel):
    info = ModelInfo(
        key="bb",
        name="Bernanke–Blanchard (2023) wage–price",
        family="Structural",
        reference="Bernanke & Blanchard (2023)",
        description=(
            "A wage–price system explaining post-pandemic inflation: a mark-up price "
            "equation (prices track wages plus energy/food shocks) and a wage Phillips "
            "curve driven by labor-market tightness (vacancy-to-unemployment ratio) and "
            "a catch-up to past inflation, both anchored to survey expectations. "
            "Forecasts headline CPI inflation by iterating the two equations forward. "
            "FRED-implementable version (omits the separate shortages term)."
        ),
        needs_activity=True,
        citation="Bernanke, B. & Blanchard, O. (2023), 'What Caused the U.S. Pandemic-Era Inflation?', Brookings/NBER w31417.",
        intuition="Wages chase prices and tightness; prices chase wages and supply shocks. Inflation is where this wage–price tug-of-war settles, pinned down by expectations.",
        unique="The only model here built for the 2021–23 surge: it separates the roles of an overheated labor market, energy/food shocks, and expectations — a live inflation-diagnosis tool.",
        strengths="Explicitly decomposes inflation into labor-market vs. supply-shock vs. expectations drivers; central banks (BoE, Banque de France) have replicated it.",
        caveats="Sample starts ~2001 (JOLTS vacancies); omits the supply-chain shortages term (not on FRED); reduced-form expectations taken from surveys rather than modeled.",
        forecast_shape="Inflation glides back toward long-run expected inflation as tightness normalizes and supply shocks fade.",
    )

    def __init__(self, n_lags: int = 2):
        super().__init__(n_lags=n_lags)
        self.n_lags = n_lags

    def _fit(self) -> None:
        import statsmodels.api as sm

        df = load_bb_data()
        self._df = df
        p = self.n_lags

        # ---- price equation, estimated in expected-inflation-gap form ----
        # (gp - Elr) = c + Σ a_i (gp_{t-i} - Elr) + b (gw - Elr) + g_e rpe + g_f rpf
        d = pd.DataFrame(index=df.index)
        d["y"] = df["gp"] - df["Elr"]
        for i in range(1, p + 1):
            d[f"gp_l{i}"] = df["gp"].shift(i) - df["Elr"]
        d["gw"] = df["gw"] - df["Elr"]
        d["rpe"] = df["rpe"]
        d["rpf"] = df["rpf"]
        dp = d.dropna()
        Xp = sm.add_constant(dp.drop(columns="y").values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._price = sm.OLS(dp["y"].values, Xp).fit()

        # ---- wage equation, estimated in expected-inflation-gap form ----
        # (gw - Ep) = c + Σ e_i (gw_{t-i} - Ep) + f vu + h (gp_{t-1} - Ep_{t-1})
        w = pd.DataFrame(index=df.index)
        w["y"] = df["gw"] - df["Ep"]
        for i in range(1, p + 1):
            w[f"gw_l{i}"] = df["gw"].shift(i) - df["Ep"]
        w["vu"] = df["vu"]
        w["catchup"] = (df["gp"].shift(1) - df["Ep"].shift(1))
        wd = w.dropna()
        Xw = sm.add_constant(wd.drop(columns="y").values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._wage = sm.OLS(wd["y"].values, Xw).fit()

        # seed the forward iteration with the most recent observations
        self._last = df.iloc[-max(p, 1) - 1:].copy()

    def _forecast(self, h: int) -> float:
        p = self.n_lags
        gp_hist = list(self._last["gp"].values)
        gw_hist = list(self._last["gw"].values)
        # exogenous drivers held at their latest values; supply shocks decay to zero
        Elr = float(self._last["Elr"].iloc[-1])
        Ep = float(self._last["Ep"].iloc[-1])
        vu = float(self._last["vu"].iloc[-1])
        rpe = float(self._last["rpe"].iloc[-1])
        rpf = float(self._last["rpf"].iloc[-1])

        pc = self._price.params
        wc = self._wage.params
        gp_prev_for_catchup = gp_hist[-1]

        for _ in range(h):
            # wage equation: (gw - Ep) = c + Σ e_i (gw_{t-i}-Ep) + f vu + h catchup
            xw = [1.0]
            for i in range(1, p + 1):
                xw.append(gw_hist[-i] - Ep)
            xw.append(vu)
            xw.append(gp_prev_for_catchup - Ep)
            gw_new = float(np.dot(wc, xw)) + Ep

            # price equation: (gp - Elr) = c + Σ a_i (gp_{t-i}-Elr) + b (gw-Elr) + g_e rpe + g_f rpf
            xp = [1.0]
            for i in range(1, p + 1):
                xp.append(gp_hist[-i] - Elr)
            xp.append(gw_new - Elr)
            xp.append(rpe)
            xp.append(rpf)
            gp_new = float(np.dot(pc, xp)) + Elr

            gp_prev_for_catchup = gp_hist[-1]
            gp_hist.append(gp_new)
            gw_hist.append(gw_new)
            rpe *= 0.5   # supply shocks dissipate
            rpf *= 0.5

        return gp_hist[-1]
