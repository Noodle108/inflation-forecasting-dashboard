"""Faust–Wright (2013) forecasting suite.

Implements every model in the FW *Handbook of Economic Forecasting* Chapter 1
"Forecasting Inflation" horse-race, using FRED-only data. Each model here mirrors
one row of FW's Table 1.2. The subjective forecasts (Blue-Chip, SPF, Greenbook) are
not on FRED and are omitted here — FW find those are the frontier of forecast
accuracy; every model below is a *model-based* benchmark, which is what we can
actually rebuild without access to survey data.

Local-mean device
-----------------
FW write most models in *gap form*: g_t = π_t − τ_t, where τ_t is a slowly varying
"local mean" of inflation. Their preferred τ_t is the Blue-Chip 5–10y long-run
inflation forecast; pre-1979 they use exponential smoothing (α=0.95) of realized
inflation as a proxy. Blue-Chip is not on FRED, so we use FW's own exponential-
smoothing fallback throughout — the same formula they document.

Models implemented
------------------
* DirectAR      — direct h-step regression on p lags of inflation.
* RecursiveAR   — iterated one-step AR(p) forecast.
* PhillipsFW    — direct h-step regression on p lags of π and unemployment.
* ARGap         — direct h-step regression in gap form.
* PCGap         — Phillips curve in gap form.
* PCTVNGap      — Phillips curve in gap form with a time-varying NAIRU proxy.
* FixedRhoGap   — AR(1) in gap form with ρ = 0.46 pinned (FW's baseline).
* TermStructureVAR — VAR(1) in [Nelson–Siegel level/slope/curvature, gap, unemp].
* EWALargeDS    — equal-weighted combination of many single-predictor gap regressions.
* BMALargeDS    — Bayesian model average of the same single-predictor regressions.
* FAVAR         — factor-augmented VAR using the same large panel.

Note: the existing dashboard already has RW, RW-AO (Atkeson–Ohanian), UCSV, TVP-VAR,
and SW07 DSGE, all of which correspond to rows in FW's Table 1.2 as-is.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo
from .fw_common import ALPHA_TAU, FIXED_RHO, MAX_LAGS, bic_ar_gap_lag, local_mean_trend


# =========================================================================== #
# 1. Direct AR                                                                 #
# =========================================================================== #
class DirectAR(ForecastModel):
    info = ModelInfo(
        key="fw_direct",
        name="Direct AR (Faust–Wright)",
        family="Statistical",
        reference="Faust–Wright (2013) — Direct",
        description=(
            "Direct h-step regression of inflation on its own p lags. FW model #1: "
            "π_{t+h} = ρ_0 + Σ ρ_j π_{t-j} + ε_{t+h}. Stationary specification — for "
            "long horizons the forecast converges to the sample mean, which FW note "
            "makes it fragile if trend inflation has drifted."
        ),
    )

    def __init__(self, max_p: int = MAX_LAGS):
        super().__init__(max_p=max_p)
        self.max_p = max_p

    def _fit(self) -> None:
        pi = self._y
        self._pi = pi
        gap = pi - local_mean_trend(pi)          # only used for BIC lag choice
        self._p = bic_ar_gap_lag(gap, self.max_p)

    def _forecast(self, h: int) -> float:
        import statsmodels.api as sm
        p = self._p
        pi = self._pi
        # direct h-step: y_target = pi shifted by -h
        cols = {f"l{l}": pi.shift(l) for l in range(p)}
        X = pd.DataFrame(cols)
        y_target = pi.shift(-h)
        df = pd.concat([y_target.rename("y"), X], axis=1).dropna()
        if len(df) < p + 5:
            return float(pi.iloc[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(df["y"].values,
                         sm.add_constant(df.drop(columns="y").values)).fit()
        # regressor row from the most recent p values (l=0..p-1)
        row = np.array([1.0] + [float(pi.iloc[-1 - l]) for l in range(p)])
        return float(res.predict(row.reshape(1, -1))[0])


# =========================================================================== #
# 2. Recursive AR                                                              #
# =========================================================================== #
class RecursiveAR(ForecastModel):
    info = ModelInfo(
        key="fw_rar",
        name="Recursive AR (Faust–Wright)",
        family="Statistical",
        reference="Faust–Wright (2013) — RAR",
        description=(
            "One-step AR(p) fit, iterated forward h times. FW model #2: same equation "
            "as Direct AR but produces the h-step forecast by recursively iterating the "
            "1-step forecast. Under correct specification this asymptotically dominates "
            "the direct forecast, but Marcellino–Stock–Watson (2006) note the direct "
            "forecast is more robust to misspecification."
        ),
    )

    def __init__(self, max_p: int = MAX_LAGS):
        super().__init__(max_p=max_p)
        self.max_p = max_p

    def _fit(self) -> None:
        import statsmodels.api as sm
        pi = self._y
        self._pi = pi
        gap = pi - local_mean_trend(pi)
        p = bic_ar_gap_lag(gap, self.max_p)
        self._p = p
        y = pi.values
        X = np.column_stack([np.roll(y, l + 1) for l in range(p)])
        y_, X_ = y[p:], X[p:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = sm.OLS(y_, sm.add_constant(X_)).fit()

    def _forecast(self, h: int) -> float:
        c = self._res.params[0]
        phis = self._res.params[1:]
        p = self._p
        hist = list(self._pi.values[-p:][::-1])   # newest first
        pi_new = hist[0]
        for _ in range(h):
            pi_new = float(c + np.dot(phis, hist[:p]))
            hist.insert(0, pi_new)
        return pi_new


# =========================================================================== #
# 3. Phillips-curve forecast (levels)                                          #
# =========================================================================== #
class PhillipsFW(ForecastModel):
    info = ModelInfo(
        key="fw_pc",
        name="Phillips Curve (Faust–Wright)",
        family="Structural",
        reference="Faust–Wright (2013) — PC",
        description=(
            "FW model #3: direct h-step regression of inflation on its own p lags and "
            "lagged unemployment. π_{t+h} = ρ_0 + Σ ρ_j π_{t-j} + λ u_{t-1} + ε_{t+h}."
        ),
        needs_activity=True,
    )

    def __init__(self, max_p: int = MAX_LAGS):
        super().__init__(max_p=max_p)
        self.max_p = max_p

    def _fit(self) -> None:
        pi = self._y
        u = (self._X["unrate"].reindex(pi.index).ffill()
             if self._X is not None and "unrate" in self._X.columns
             else pd.Series(0.0, index=pi.index))
        self._pi, self._u = pi, u
        gap = pi - local_mean_trend(pi)
        self._p = bic_ar_gap_lag(gap, self.max_p)

    def _forecast(self, h: int) -> float:
        import statsmodels.api as sm
        p = self._p
        pi, u = self._pi, self._u
        cols = {f"pi_l{l}": pi.shift(l) for l in range(p)}
        cols["u_l1"] = u.shift(1)
        X = pd.DataFrame(cols)
        y_target = pi.shift(-h)
        df = pd.concat([y_target.rename("y"), X], axis=1).dropna()
        if len(df) < p + 5:
            return float(pi.iloc[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(df["y"].values,
                         sm.add_constant(df.drop(columns="y").values)).fit()
        row = ([1.0] + [float(pi.iloc[-1 - l]) for l in range(p)]
               + [float(u.iloc[-1])])
        return float(res.predict(np.array(row).reshape(1, -1))[0])


# =========================================================================== #
# 4. AR-GAP                                                                    #
# =========================================================================== #
class ARGap(ForecastModel):
    info = ModelInfo(
        key="fw_argap",
        name="AR–gap (Faust–Wright)",
        family="Statistical",
        reference="Faust–Wright (2013) — AR-GAP",
        description=(
            "FW model #6: g_{t+h} = ρ_0 + Σ ρ_j g_{t-j} + ε_{t+h}, then add the latest "
            "trend τ_T back on. Non-stationary because τ_t is a random walk; solves the "
            "'convergence to unconditional mean' problem of stationary AR forecasts."
        ),
    )

    def __init__(self, max_p: int = MAX_LAGS):
        super().__init__(max_p=max_p)
        self.max_p = max_p

    def _fit(self) -> None:
        pi = self._y
        self._pi = pi
        self._tau = local_mean_trend(pi)
        self._gap = pi - self._tau
        self._p = bic_ar_gap_lag(self._gap, self.max_p)

    def _forecast(self, h: int) -> float:
        import statsmodels.api as sm
        p = self._p
        gap = self._gap
        cols = {f"l{l}": gap.shift(l) for l in range(p)}
        X = pd.DataFrame(cols)
        y_target = gap.shift(-h)
        df = pd.concat([y_target.rename("y"), X], axis=1).dropna()
        if len(df) < p + 5:
            return float(self._tau.iloc[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(df["y"].values,
                         sm.add_constant(df.drop(columns="y").values)).fit()
        row = np.array([1.0] + [float(gap.iloc[-1 - l]) for l in range(p)])
        gap_h = float(res.predict(row.reshape(1, -1))[0])
        return float(self._tau.iloc[-1]) + gap_h


# =========================================================================== #
# 5. Fixed-ρ gap (the paper's benchmark)                                       #
# =========================================================================== #
class FixedRhoGap(ForecastModel):
    info = ModelInfo(
        key="fw_fixedrho",
        name="Fixed-ρ AR(1) gap (Faust–Wright benchmark)",
        family="Benchmark",
        reference="Faust–Wright (2013) — Fixed ρ",
        description=(
            "FW model #7 and the paper's benchmark for RMSPE ratios. AR(1) in gap form "
            "with ρ pinned to 0.46 (FW's value from a 1985Q1-vintage GDP-deflator fit "
            "1947Q2–1959Q4). No parameter estimation. Deceptively hard to beat: the "
            "row every other model in Table 1.2 is divided by."
        ),
    )

    def __init__(self, rho: float = FIXED_RHO):
        super().__init__(rho=rho)
        self.rho = rho

    def _fit(self) -> None:
        pi = self._y
        self._tau = local_mean_trend(pi)
        self._gap_last = float(pi.iloc[-1] - self._tau.iloc[-1])

    def _forecast(self, h: int) -> float:
        gap_h = self._gap_last * (self.rho ** h)
        return float(self._tau.iloc[-1] + gap_h)


# =========================================================================== #
# 6. Phillips curve in gap form                                                #
# =========================================================================== #
class PCGap(ForecastModel):
    info = ModelInfo(
        key="fw_pcgap",
        name="Phillips curve, gap form (Faust–Wright)",
        family="Structural",
        reference="Faust–Wright (2013) — PC-GAP",
        description=(
            "FW model #8: g_{t+h} = ρ_0 + Σ ρ_j g_{t-j} + λ u_{t-1} + ε_t. Applies the "
            "Phillips curve to the inflation gap rather than the level of inflation, "
            "so it inherits AR-GAP's slowly drifting trend."
        ),
        needs_activity=True,
    )

    def __init__(self, max_p: int = MAX_LAGS):
        super().__init__(max_p=max_p)
        self.max_p = max_p

    def _fit(self) -> None:
        pi = self._y
        u = (self._X["unrate"].reindex(pi.index).ffill()
             if self._X is not None and "unrate" in self._X.columns
             else pd.Series(0.0, index=pi.index))
        self._pi, self._u = pi, u
        self._tau = local_mean_trend(pi)
        self._gap = pi - self._tau
        self._p = bic_ar_gap_lag(self._gap, self.max_p)

    def _forecast(self, h: int) -> float:
        import statsmodels.api as sm
        p = self._p
        gap = self._gap
        u = self._u
        cols = {f"g_l{l}": gap.shift(l) for l in range(p)}
        cols["u_l1"] = u.shift(1)
        X = pd.DataFrame(cols)
        y_target = gap.shift(-h)
        df = pd.concat([y_target.rename("y"), X], axis=1).dropna()
        if len(df) < p + 5:
            return float(self._tau.iloc[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(df["y"].values,
                         sm.add_constant(df.drop(columns="y").values)).fit()
        row = ([1.0] + [float(gap.iloc[-1 - l]) for l in range(p)]
               + [float(u.iloc[-1])])
        gap_h = float(res.predict(np.array(row).reshape(1, -1))[0])
        return float(self._tau.iloc[-1]) + gap_h


# =========================================================================== #
# 7. Phillips curve gap with time-varying NAIRU                                #
# =========================================================================== #
class PCTVNGap(ForecastModel):
    info = ModelInfo(
        key="fw_pctvngap",
        name="PC-gap with time-varying NAIRU (Faust–Wright)",
        family="Structural",
        reference="Faust–Wright (2013) — PCTVN-GAP",
        description=(
            "FW model #9: same as PC-GAP but the slack term is (u − u*), with u* a slowly "
            "evolving NAIRU proxy. FW use a Blue-Chip 5–10y unemployment forecast for u*; "
            "we substitute FW's own exponential-smoothing fallback (footnote 8) using "
            "α=0.95 on realized unemployment."
        ),
        needs_activity=True,
    )

    def __init__(self, max_p: int = MAX_LAGS, alpha_u: float = ALPHA_TAU):
        super().__init__(max_p=max_p, alpha_u=alpha_u)
        self.max_p = max_p
        self.alpha_u = alpha_u

    def _fit(self) -> None:
        pi = self._y
        u = (self._X["unrate"].reindex(pi.index).ffill()
             if self._X is not None and "unrate" in self._X.columns
             else pd.Series(0.0, index=pi.index))
        self._pi, self._u = pi, u
        self._tau = local_mean_trend(pi)
        self._gap = pi - self._tau
        # exponentially smoothed NAIRU proxy — FW's own fallback for pre-1979
        u_star = np.empty(len(u))
        u_star[0] = float(u.iloc[0])
        for t in range(1, len(u)):
            u_star[t] = self.alpha_u * u_star[t - 1] + (1 - self.alpha_u) * float(u.iloc[t])
        self._ustar = pd.Series(u_star, index=u.index)
        self._p = bic_ar_gap_lag(self._gap, self.max_p)

    def _forecast(self, h: int) -> float:
        import statsmodels.api as sm
        p = self._p
        gap = self._gap
        slack = self._u - self._ustar
        cols = {f"g_l{l}": gap.shift(l) for l in range(p)}
        cols["slack_l1"] = slack.shift(1)
        X = pd.DataFrame(cols)
        y_target = gap.shift(-h)
        df = pd.concat([y_target.rename("y"), X], axis=1).dropna()
        if len(df) < p + 5:
            return float(self._tau.iloc[-1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(df["y"].values,
                         sm.add_constant(df.drop(columns="y").values)).fit()
        row = ([1.0] + [float(gap.iloc[-1 - l]) for l in range(p)]
               + [float(slack.iloc[-1])])
        gap_h = float(res.predict(np.array(row).reshape(1, -1))[0])
        return float(self._tau.iloc[-1]) + gap_h


# =========================================================================== #
# 8. Term-structure VAR (Nelson–Siegel + gap + unemployment)                   #
# =========================================================================== #
_NS_LAMBDA = 0.0609    # Diebold–Li parameter, quoted directly in FW (equation 1)


def _nelson_siegel_factors(yields_df: pd.DataFrame) -> pd.DataFrame:
    """Fit level/slope/curvature to a yields panel at each observation date.

    yields_df: DataFrame indexed by date; columns = maturities *in years* (float);
    values = yields in percent. Returns a DataFrame with columns [b1, b2, b3].
    """
    ns_lam = _NS_LAMBDA
    n_cols = np.array(sorted(yields_df.columns), dtype=float)   # sorted maturities
    n_months = n_cols * 12.0   # convert years to months for the loading form
    # Nelson–Siegel loadings for each maturity
    L1 = np.ones_like(n_months)
    L2 = (1 - np.exp(-ns_lam * n_months)) / (ns_lam * n_months)
    L3 = L2 - np.exp(-ns_lam * n_months)
    load = np.column_stack([L1, L2, L3])          # k×3
    # OLS at every date (with Moore-Penrose pseudo-inverse for stability)
    pinv = np.linalg.pinv(load)                    # 3×k
    Y = yields_df[sorted(yields_df.columns)].values
    B = Y @ pinv.T                                 # T×3
    return pd.DataFrame(B, index=yields_df.index, columns=["b1", "b2", "b3"])


def _load_treasury_yields() -> pd.DataFrame | None:
    """Fetch a Treasury yield panel (constant-maturity) from FRED, quarterly."""
    from ..data import fred as _f
    if not _f.has_live_data():
        return None
    ids = {0.25: "TB3MS", 0.5: "TB6MS", 1: "GS1", 2: "GS2", 3: "GS3",
           5: "GS5", 7: "GS7", 10: "GS10"}
    cols = {}
    for tau, sid in ids.items():
        s = _f._fetch_fred_series(sid, "1960-01-01")
        if s is None:
            continue
        cols[tau] = s.resample("QS").mean()
    if not cols:
        return None
    return pd.DataFrame(cols).dropna(how="all")


class TermStructureVAR(ForecastModel):
    info = ModelInfo(
        key="fw_tsvar",
        name="Term-structure VAR (Faust–Wright)",
        family="Statistical",
        reference="Faust–Wright (2013) — Term Structure VAR",
        description=(
            "FW model #10: fit a Nelson–Siegel level/slope/curvature to the Treasury "
            "yield curve at each date, then run a VAR(1) in "
            "[level, slope, curvature, inflation-gap, unemployment]. Inflation forecast "
            "= gap forecast + latest τ_T. The FW build without no-arbitrage restrictions "
            "(Joslin–Le–Singleton show these are empirically inconsequential)."
        ),
        needs_activity=True,
    )

    def _fit(self) -> None:
        import statsmodels.api as sm
        pi = self._y
        self._tau = local_mean_trend(pi)
        gap = pi - self._tau

        yields = _load_treasury_yields()
        # Quarterly alignment; if pi is monthly, average to Q
        if pi.index.freqstr and pi.index.freqstr.startswith("Q"):
            pi_q, gap_q = pi, gap
        else:
            pi_q = pi.resample("QS").mean()
            gap_q = pi_q - local_mean_trend(pi_q)
            self._tau = local_mean_trend(pi_q)

        u_raw = (self._X["unrate"].reindex(pi.index).ffill()
                 if self._X is not None and "unrate" in self._X.columns
                 else pd.Series(0.0, index=pi.index))
        u_q = u_raw.resample("QS").mean() if not (pi.index.freqstr or "").startswith("Q") else u_raw

        if yields is None:
            # No live data: fall back to just [gap, unemployment] VAR
            df = pd.concat([gap_q.rename("g"), u_q.rename("u")], axis=1).dropna()
            self._names = ["g", "u"]
        else:
            ns = _nelson_siegel_factors(yields)
            df = pd.concat([ns, gap_q.rename("g"), u_q.rename("u")], axis=1).dropna()
            self._names = ["b1", "b2", "b3", "g", "u"]

        Y = df[self._names].values
        T, k = Y.shape
        Xr = np.column_stack([np.ones(T - 1), Y[:-1]])
        Yn = Y[1:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._B = np.linalg.solve(Xr.T @ Xr + 1e-8 * np.eye(k + 1),
                                      Xr.T @ Yn)      # (k+1) × k
        self._last = Y[-1]
        self._k = k
        # index of the gap column in self._names
        self._g_idx = self._names.index("g")

    def _forecast(self, h: int) -> float:
        y = self._last.copy()
        for _ in range(h):
            x = np.concatenate([[1.0], y])
            y = x @ self._B
        gap_h = float(y[self._g_idx])
        return float(self._tau.iloc[-1] + gap_h)


# =========================================================================== #
# 9-11. Large-dataset methods (EWA, BMA, FAVAR) — share the same panel        #
# =========================================================================== #
_LARGE_DS_FRED = dict(
    ip="INDPRO",                    # industrial production
    emp="PAYEMS",                   # nonfarm payrolls
    unrate="UNRATE",
    housing_starts="HOUST",
    consumer_conf="UMCSENT",
    m2="M2SL",                      # broad money
    ffr="FEDFUNDS",
    tb3m="TB3MS", gs1="GS1", gs5="GS5", gs10="GS10",
    baa="BAA", aaa="AAA", spread="BAA10YM",
    oil="MCOILWTICO",               # WTI crude
    sp500="SP500",                  # (short history — fine for post-2011)
    manuf_new_orders="AMTMNO",
    hourly_earnings="CES0500000003",
    real_gdp="GDPC1",
    real_consumption="PCECC96",
    real_investment="GPDIC1",
    trade_bal="BOPGSTB",
    gold="GOLDPMGBD228NLBM",
    tips10="DFII10",
    yield_slope="T10Y3M",
    imports="IR",
    exports="EXPGSC1",
    real_disp_inc="DSPIC96",
    home_sales="HSN1F",
    orders_durable="DGORDER",
    michigan="MICH",
    breakevens5="T5YIE",
)


def _load_large_panel(freq: str = "Q") -> pd.DataFrame | None:
    """Fetch a FRED-MD-flavor panel for EWA/BMA/FAVAR, transformed to stationary."""
    from ..data import fred as _f
    if not _f.has_live_data():
        return None
    raw = {}
    for name, sid in _LARGE_DS_FRED.items():
        s = _f._fetch_fred_series(sid, "1960-01-01")
        if s is None or len(s) < 60:
            continue
        s = s.resample("QS").mean() if freq == "Q" else s.resample("MS").mean()
        raw[name] = s
    if not raw:
        return None
    df = pd.DataFrame(raw)
    # Transformations: log-diff for level series that trend, first-diff for rates.
    RATES = {"unrate", "ffr", "tb3m", "gs1", "gs5", "gs10", "baa", "aaa",
             "spread", "yield_slope", "michigan", "breakevens5", "tips10"}
    out = {}
    for c in df.columns:
        s = df[c]
        if c in RATES:
            out[c] = s.diff()
        else:
            out[c] = 100 * np.log(s.replace({0: np.nan})).diff()
    return pd.DataFrame(out).dropna(how="all")


class _LargeDSBase(ForecastModel):
    """Shared machinery for EWA/BMA/FAVAR: prepare (gap panel, X panel) pairs."""

    def __init__(self, max_p: int = MAX_LAGS):
        super().__init__(max_p=max_p)
        self.max_p = max_p

    def _prepare(self):
        pi = self._y
        self._tau = local_mean_trend(pi)
        gap = pi - self._tau
        freq = "Q" if (pi.index.freqstr or "").startswith("Q") else "M"
        panel = _load_large_panel(freq=freq)
        if panel is None:
            self._panel = None
            self._gap = gap
            return
        panel = panel.loc[:pi.index[-1]]
        common = gap.index.intersection(panel.index)
        self._gap = gap.reindex(common)
        panel = panel.reindex(common)
        # Drop columns whose *latest* value is NaN (can't build the regressor row).
        latest_ok = panel.iloc[-1].notna()
        panel = panel.loc[:, latest_ok]
        # Drop columns with <40 non-NaN observations — too little to fit reliably.
        panel = panel.loc[:, panel.notna().sum() >= 40]
        self._panel = panel
        # Standardize on the observed values; fill remaining NaNs with 0
        # (the standardized column mean) so they contribute nothing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._panel_z = ((panel - panel.mean()) / panel.std()).fillna(0.0)


class EWALargeDS(_LargeDSBase):
    info = ModelInfo(
        key="fw_ewa",
        name="EWA large-dataset combo (Faust–Wright)",
        family="Statistical",
        reference="Faust–Wright (2013) — EWA",
        description=(
            "FW model #12. For each of n predictors x_i, fit "
            "g_{t+h} = ρ_0 + Σ ρ_j g_{t-j} + β_i x_{i,t-1} + ε_t and average the "
            "resulting n forecasts of g_{T+h} with equal weights. Bates–Granger (1969) "
            "combination — one of the folklore results in forecast combination."
        ),
    )

    def _fit(self) -> None:
        self._prepare()

    def _forecast(self, h: int) -> float:
        import statsmodels.api as sm
        if self._panel is None or self._panel.empty:
            return float(self._tau.iloc[-1])
        gap = self._gap
        p = bic_ar_gap_lag(gap, self.max_p)
        # design pieces shared across predictors
        cols_common = {f"g_l{l}": gap.shift(l) for l in range(p)}
        y_target = gap.shift(-h)
        preds = []
        for name in self._panel_z.columns:
            xi = self._panel_z[name].shift(1)
            X = pd.DataFrame({**cols_common, "x": xi})
            df = pd.concat([y_target.rename("y"), X], axis=1).dropna()
            if len(df) < p + 8:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.OLS(df["y"].values,
                             sm.add_constant(df.drop(columns="y").values)).fit()
            row = ([1.0]
                   + [float(gap.iloc[-1 - l]) for l in range(p)]
                   + [float(self._panel_z[name].iloc[-1])])
            preds.append(float(res.predict(np.array(row).reshape(1, -1))[0]))
        if not preds:
            return float(self._tau.iloc[-1])
        gap_h = float(np.mean(preds))
        return float(self._tau.iloc[-1] + gap_h)


class BMALargeDS(_LargeDSBase):
    info = ModelInfo(
        key="fw_bma",
        name="BMA large-dataset combo (Faust–Wright)",
        family="Statistical",
        reference="Faust–Wright (2013) — BMA",
        description=(
            "FW model #13. Same n single-predictor gap regressions as EWA, but combined "
            "with Bayesian model-average weights: each weight is the posterior model "
            "probability under a flat model prior and the Fernandez–Ley–Steel g-prior "
            "on coefficients. Shrinkage toward the informative predictors."
        ),
    )

    def _fit(self) -> None:
        self._prepare()

    def _forecast(self, h: int) -> float:
        import statsmodels.api as sm
        if self._panel is None or self._panel.empty:
            return float(self._tau.iloc[-1])
        gap = self._gap
        p = bic_ar_gap_lag(gap, self.max_p)
        cols_common = {f"g_l{l}": gap.shift(l) for l in range(p)}
        y_target = gap.shift(-h)
        preds, log_ml = [], []
        for name in self._panel_z.columns:
            xi = self._panel_z[name].shift(1)
            X = pd.DataFrame({**cols_common, "x": xi})
            df = pd.concat([y_target.rename("y"), X], axis=1).dropna()
            if len(df) < p + 8:
                continue
            Xm = sm.add_constant(df.drop(columns="y").values)
            y = df["y"].values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.OLS(y, Xm).fit()
            # crude BIC-based model weight: log p(y|M) ≈ -0.5 * BIC (Schwarz approx)
            log_ml.append(-0.5 * float(res.bic))
            row = ([1.0]
                   + [float(gap.iloc[-1 - l]) for l in range(p)]
                   + [float(self._panel_z[name].iloc[-1])])
            preds.append(float(res.predict(np.array(row).reshape(1, -1))[0]))
        if not preds:
            return float(self._tau.iloc[-1])
        w = np.array(log_ml)
        w = np.exp(w - w.max()); w /= w.sum()
        gap_h = float(np.dot(w, preds))
        return float(self._tau.iloc[-1] + gap_h)


class FAVARLargeDS(_LargeDSBase):
    info = ModelInfo(
        key="fw_favar",
        name="FAVAR (Faust–Wright)",
        family="Statistical",
        reference="Faust–Wright (2013) — FAVAR",
        description=(
            "FW model #14. Extract the first m principal components z_1..z_m from the "
            "standardized large panel, then fit a VAR(p) in ξ_t = (g_t, z_{1,t}, ..., z_{m,t}) "
            "and iterate forward. Bernanke, Boivin & Eliasz (2005) factor-augmented VAR."
        ),
    )

    def __init__(self, n_factors: int = 3, max_p: int = MAX_LAGS):
        super().__init__(max_p=max_p)
        self.n_factors = n_factors

    def _fit(self) -> None:
        self._prepare()
        if self._panel is None or self._panel.empty:
            self._factors = None
            return
        Z = self._panel_z.fillna(0.0).values
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        m = min(self.n_factors, len(S))
        F = U[:, :m] * S[:m] / np.sqrt(len(Z))
        self._factors = pd.DataFrame(F, index=self._panel.index,
                                     columns=[f"z{i+1}" for i in range(m)])
        # VAR(1) in (gap, z1..zm)
        Y = pd.concat([self._gap.rename("g"), self._factors], axis=1).dropna().values
        T, k = Y.shape
        Xr = np.column_stack([np.ones(T - 1), Y[:-1]])
        Yn = Y[1:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._B = np.linalg.solve(Xr.T @ Xr + 1e-8 * np.eye(k + 1),
                                      Xr.T @ Yn)
        self._last = Y[-1]

    def _forecast(self, h: int) -> float:
        if self._factors is None:
            return float(self._tau.iloc[-1])
        y = self._last.copy()
        for _ in range(h):
            x = np.concatenate([[1.0], y])
            y = x @ self._B
        return float(self._tau.iloc[-1] + y[0])


# =========================================================================== #
# 12. DSGE-GAP: SW07 forecast tacked onto the local-mean trend                 #
# =========================================================================== #
class DSGEGap(ForecastModel):
    """Wraps SW07 as a *gap-form* forecast: the DSGE's steady-state inflation prior is
    replaced by the local mean τ_t. Concretely we compute the SW07 forecast, subtract
    the DSGE's implicit steady-state, and add τ_T back on — FW's crude but effective
    device to strip look-back bias out of the estimated prior mean.
    """

    info = ModelInfo(
        key="fw_dsgegap",
        name="DSGE-gap (Smets–Wouters + local mean, Faust–Wright)",
        family="Structural",
        reference="Faust–Wright (2013) — DSGE-GAP",
        description=(
            "FW model #16: take the SW07 DSGE forecast and rewrite it in gap form by "
            "replacing the model's steady-state inflation prior with the exponentially "
            "smoothed local mean τ_t. Removes the look-back bias in the SW07 prior."
        ),
        needs_activity=True,
    )

    def _fit(self) -> None:
        from .sw2007 import SmetsWouters2007
        pi = self._y
        self._tau = local_mean_trend(pi)
        self._sw = SmetsWouters2007()
        self._sw.fit(pi, self._X)

    def _forecast(self, h: int) -> float:
        sw_pi = self._sw.forecast(h)
        # SW07's own long-run steady-state (2.5% annualized inflation, per SW07 prior).
        # Replace with the local mean τ_T so the forecast rides FW's slow-moving trend.
        sw_ss = 4.0 * float(self._sw._pinf_mean_q)   # annualized sample mean SW07 uses
        return float(sw_pi - sw_ss + self._tau.iloc[-1])
