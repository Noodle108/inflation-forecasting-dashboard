"""Full 7-shock Smets–Wouters (2007) DSGE.

This implements the complete Smets & Wouters (2007, AER) medium-scale New Keynesian
DSGE: a sticky-price/sticky-wage economy with a parallel flexible-price economy (for the
welfare-relevant output gap), capital with variable utilization and investment
adjustment costs, external habit in consumption, Calvo pricing and wage-setting with
indexation, and a Taylor-type monetary rule. It is driven by the **seven structural
shocks** of the paper:

    productivity (a), risk premium (b), government spending (g),
    investment-specific (qs), monetary policy (ms),
    price mark-up (spinf, ARMA), wage mark-up (sw, ARMA).

Deep parameters are fixed at the paper's estimated **posterior mode**. The linear
rational-expectations equilibrium is solved with `gensys` (validated separately against
the small-NK analytic solution). The model is then Kalman-filtered on the seven standard
US observables from FRED and projected forward to forecast inflation.

The ~33 log-linearized equations are entered through a small symbolic builder
(`_REBuilder`) that assembles the gensys matrices and auto-creates the expectational
variables, which keeps the transcription readable and checkable.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo
from .gensys import gensys


# --------------------------------------------------------------------------- #
# Symbolic linear-RE builder
# --------------------------------------------------------------------------- #
class _REBuilder:
    """Assemble gensys matrices from equations written as term lists.

    Each equation is `sum(terms) + sum(shock terms) = 0`, where a term is
    (variable, time-shift ∈ {-1,0,+1}, coefficient) and a shock term is
    (shock_name, coefficient). Any variable used at shift +1 automatically gets an
    expectational companion E_<var> and the identity var_t = E_<var>_{t-1} + eta.
    """

    def __init__(self, varnames, shocknames):
        self.varnames = list(varnames)
        self.shocknames = list(shocknames)
        self.eqs = []

    def add(self, terms, shocks=None):
        self.eqs.append((terms, shocks or []))

    def build(self):
        fwd = sorted({v for (terms, _) in self.eqs for (v, s, _c) in terms if s == 1})
        evars = ["E_" + v for v in fwd]
        allvars = self.varnames + evars
        vidx = {v: i for i, v in enumerate(allvars)}
        sidx = {s: i for i, s in enumerate(self.shocknames)}
        n, neta, nz = len(allvars), len(fwd), len(self.shocknames)

        assert len(self.eqs) == len(self.varnames), (
            f"{len(self.eqs)} equations for {len(self.varnames)} base variables")

        g0 = np.zeros((n, n)); g1 = np.zeros((n, n))
        psi = np.zeros((n, nz)); pi = np.zeros((n, neta))
        row = 0
        for (terms, shocks) in self.eqs:
            for (v, s, c) in terms:
                if s == 0:
                    g0[row, vidx[v]] += c
                elif s == 1:
                    g0[row, vidx["E_" + v]] += c
                elif s == -1:
                    g1[row, vidx[v]] += -c
                else:
                    raise ValueError("shift must be -1, 0, or +1")
            # shock terms are written as they appear additively on the RHS
            # (e.g. a_t = rho*a_{t-1} + ea  →  shocks=[("ea", 1)]).
            for (sh, c) in shocks:
                psi[row, sidx[sh]] += c
            row += 1
        for k, v in enumerate(fwd):
            g0[row, vidx[v]] = 1.0
            g1[row, vidx["E_" + v]] = 1.0
            pi[row, k] = 1.0
            row += 1
        return g0, g1, psi, pi, allvars


# --------------------------------------------------------------------------- #
# Parameters (Smets–Wouters 2007 posterior mode) and the model matrices
# --------------------------------------------------------------------------- #
SW_PARAMS = dict(
    # calibrated
    ctou=0.025, clandaw=1.5, cg=0.18, curvp=10.0, curvw=10.0,
    # estimated deep parameters (posterior mode, SW07 Table 1)
    calfa=0.19, csigma=1.38, cfc=1.61, cgy=0.51,
    csadjcost=5.74, chabb=0.71, cprobw=0.70, csigl=1.83, cprobp=0.66,
    cindw=0.58, cindp=0.24, czcap=0.54,
    crpi=2.04, crr=0.81, cry=0.08, crdy=0.22,
    constepinf=0.81, constebeta=0.16, ctrend=0.43,
    # shock persistence / MA
    crhoa=0.95, crhob=0.22, crhog=0.97, crhoqs=0.71, crhoms=0.15,
    crhopinf=0.89, crhow=0.96, cmap=0.69, cmaw=0.84,
    # shock standard deviations
    sig_a=0.45, sig_b=0.23, sig_g=0.53, sig_qs=0.45, sig_m=0.24,
    sig_pinf=0.14, sig_w=0.24,
)

# order of the seven exogenous innovations
SHOCKS = ["ea", "eb", "eg", "eqs", "em", "epinf", "ew"]


def _derived(p):
    d = dict(p)
    d["cpie"] = 1 + p["constepinf"] / 100
    d["cgamma"] = 1 + p["ctrend"] / 100
    d["cbeta"] = 1 / (1 + p["constebeta"] / 100)
    d["cbetabar"] = d["cbeta"] * d["cgamma"] ** (-p["csigma"])
    d["crk"] = (d["cbeta"] ** (-1)) * (d["cgamma"] ** p["csigma"]) - (1 - p["ctou"])
    d["clandap"] = p["cfc"]
    d["cw_ss"] = (p["calfa"] ** p["calfa"] * (1 - p["calfa"]) ** (1 - p["calfa"]) /
                  (d["clandap"] * d["crk"] ** p["calfa"])) ** (1 / (1 - p["calfa"]))
    d["cikbar"] = 1 - (1 - p["ctou"]) / d["cgamma"]
    d["cik"] = d["cikbar"] * d["cgamma"]
    d["clk"] = ((1 - p["calfa"]) / p["calfa"]) * (d["crk"] / d["cw_ss"])
    d["cky"] = p["cfc"] * d["clk"] ** (p["calfa"] - 1)
    d["ciy"] = d["cik"] * d["cky"]
    d["ccy"] = 1 - p["cg"] - d["ciy"]
    d["crkky"] = d["crk"] * d["cky"]
    d["cwhlc"] = (1 / p["clandaw"]) * (1 - p["calfa"]) / p["calfa"] * d["crk"] * d["cky"] / d["ccy"]
    return d


def build_sw_model(params=None):
    """Return (g0, g1, psi, pi, varlist) for the SW07 model."""
    p = _derived(params or SW_PARAMS)
    (ctou, calfa, csigma, cfc, cgy, csadj, chabb, cprobw, csigl, cprobp,
     cindw, cindp, czcap, crpi, crr, cry, crdy, clandaw, curvp, curvw) = (
        p["ctou"], p["calfa"], p["csigma"], p["cfc"], p["cgy"], p["csadjcost"],
        p["chabb"], p["cprobw"], p["csigl"], p["cprobp"], p["cindw"], p["cindp"],
        p["czcap"], p["crpi"], p["crr"], p["cry"], p["crdy"], p["clandaw"],
        p["curvp"], p["curvw"])
    cgamma, cbetabar, crk = p["cgamma"], p["cbetabar"], p["crk"]
    ccy, ciy, crkky, cwhlc, crhoa = p["ccy"], p["ciy"], p["crkky"], p["cwhlc"], p["crhoa"]

    hgc = chabb / cgamma
    a_inv = 1 / (1 + cbetabar * cgamma)
    zc = (1 - czcap) / czcap
    kt1 = crk / (crk + (1 - ctou))
    kt2 = (1 - ctou) / (crk + (1 - ctou))
    bpk = csigma * (1 + hgc) / (1 - hgc)             # risk-premium loading in q-equation
    c1 = hgc / (1 + hgc); c2 = 1 / (1 + hgc)
    c3 = ((csigma - 1) * cwhlc) / (csigma * (1 + hgc))
    c4 = (1 - hgc) / (csigma * (1 + hgc))
    invpk = 1 / (cgamma ** 2 * csadj)
    f = 1 + cbetabar * cgamma
    d1 = 1 + cbetabar * cgamma * cindp
    cpc = ((1 - cprobp) * (1 - cbetabar * cgamma * cprobp) / cprobp) / ((cfc - 1) * curvp + 1)
    e1, e2, e3 = 1 / f, cbetabar * cgamma / f, cindw / f
    e4 = (1 + cbetabar * cgamma * cindw) / f
    e5 = cbetabar * cgamma / f
    e6 = ((1 - cprobw) * (1 - cbetabar * cgamma * cprobw) / (f * cprobw)) * (1 / ((clandaw - 1) * curvw + 1))
    ckpqs = p["cikbar"] * cgamma ** 2 * csadj

    flex = ["labf", "kf", "kpf", "yf", "cf", "invef", "pkf", "wf", "rkf", "zcapf", "rrf"]
    stick = ["mc", "zcap", "rk", "k", "kp", "inve", "pk", "c", "y", "lab", "pinf", "w", "r"]
    exog = ["a", "b", "g", "qs", "ms", "spinf", "sw", "epinfma", "ewma"]
    B = _REBuilder(flex + stick + exog, SHOCKS)

    # ---- flexible-price economy ----
    B.add([("rkf", 0, calfa), ("wf", 0, 1 - calfa), ("a", 0, -1)])            # mcf = 0
    B.add([("zcapf", 0, 1), ("rkf", 0, -zc)])
    B.add([("rkf", 0, 1), ("wf", 0, -1), ("labf", 0, -1), ("kf", 0, 1)])
    B.add([("kf", 0, 1), ("kpf", -1, -1), ("zcapf", 0, -1)])
    B.add([("invef", 0, 1), ("invef", -1, -a_inv), ("invef", 1, -a_inv * cbetabar * cgamma),
           ("pkf", 0, -a_inv * invpk), ("qs", 0, -1)])
    B.add([("pkf", 0, 1), ("rrf", 0, 1), ("b", 0, -bpk),
           ("rkf", 1, -kt1), ("pkf", 1, -kt2)])
    B.add([("cf", 0, 1), ("cf", -1, -c1), ("cf", 1, -c2),
           ("labf", 0, -c3), ("labf", 1, c3), ("rrf", 0, c4), ("b", 0, -1)])
    B.add([("yf", 0, 1), ("cf", 0, -ccy), ("invef", 0, -ciy), ("g", 0, -1), ("zcapf", 0, -crkky)])
    B.add([("yf", 0, 1), ("kf", 0, -cfc * calfa), ("labf", 0, -cfc * (1 - calfa)), ("a", 0, -cfc)])
    B.add([("wf", 0, 1), ("labf", 0, -csigl), ("cf", 0, -1 / (1 - hgc)), ("cf", -1, hgc / (1 - hgc))])
    B.add([("kpf", 0, 1), ("kpf", -1, -(1 - p["cikbar"])), ("invef", 0, -p["cikbar"]),
           ("qs", 0, -ckpqs)])

    # ---- sticky-price economy ----
    B.add([("mc", 0, 1), ("rk", 0, -calfa), ("w", 0, -(1 - calfa)), ("a", 0, 1)])
    B.add([("zcap", 0, 1), ("rk", 0, -zc)])
    B.add([("rk", 0, 1), ("w", 0, -1), ("lab", 0, -1), ("k", 0, 1)])
    B.add([("k", 0, 1), ("kp", -1, -1), ("zcap", 0, -1)])
    B.add([("inve", 0, 1), ("inve", -1, -a_inv), ("inve", 1, -a_inv * cbetabar * cgamma),
           ("pk", 0, -a_inv * invpk), ("qs", 0, -1)])
    B.add([("pk", 0, 1), ("r", 0, 1), ("pinf", 1, -1), ("b", 0, -bpk),
           ("rk", 1, -kt1), ("pk", 1, -kt2)])
    B.add([("c", 0, 1), ("c", -1, -c1), ("c", 1, -c2),
           ("lab", 0, -c3), ("lab", 1, c3),
           ("r", 0, c4), ("pinf", 1, -c4), ("b", 0, -1)])
    B.add([("y", 0, 1), ("c", 0, -ccy), ("inve", 0, -ciy), ("g", 0, -1), ("zcap", 0, -crkky)])
    B.add([("y", 0, 1), ("k", 0, -cfc * calfa), ("lab", 0, -cfc * (1 - calfa)), ("a", 0, -cfc)])
    B.add([("pinf", 0, 1), ("pinf", 1, -cbetabar * cgamma / d1), ("pinf", -1, -cindp / d1),
           ("mc", 0, -cpc / d1), ("spinf", 0, -1)])
    B.add([("w", 0, 1 + e6), ("w", -1, -e1), ("w", 1, -e2),
           ("pinf", -1, -e3), ("pinf", 0, e4), ("pinf", 1, -e5),
           ("lab", 0, -e6 * csigl), ("c", 0, -e6 / (1 - hgc)), ("c", -1, e6 * hgc / (1 - hgc)),
           ("sw", 0, -1)])
    B.add([("r", 0, 1), ("pinf", 0, -crpi * (1 - crr)),
           ("y", 0, -(cry * (1 - crr) + crdy)), ("yf", 0, (cry * (1 - crr) + crdy)),
           ("y", -1, crdy), ("yf", -1, -crdy), ("r", -1, -crr), ("ms", 0, -1)])
    B.add([("kp", 0, 1), ("kp", -1, -(1 - p["cikbar"])), ("inve", 0, -p["cikbar"]),
           ("qs", 0, -ckpqs)])

    # ---- exogenous shock processes ----
    B.add([("a", 0, 1), ("a", -1, -crhoa)], [("ea", 1)])
    B.add([("b", 0, 1), ("b", -1, -p["crhob"])], [("eb", 1)])
    B.add([("g", 0, 1), ("g", -1, -p["crhog"])], [("eg", 1), ("ea", cgy)])
    B.add([("qs", 0, 1), ("qs", -1, -p["crhoqs"])], [("eqs", 1)])
    B.add([("ms", 0, 1), ("ms", -1, -p["crhoms"])], [("em", 1)])
    B.add([("epinfma", 0, 1)], [("epinf", 1)])
    B.add([("spinf", 0, 1), ("spinf", -1, -p["crhopinf"]), ("epinfma", 0, -1), ("epinfma", -1, p["cmap"])])
    B.add([("ewma", 0, 1)], [("ew", 1)])
    B.add([("sw", 0, 1), ("sw", -1, -p["crhow"]), ("ewma", 0, -1), ("ewma", -1, p["cmaw"])])

    g0, g1, psi, pi, varlist = B.build()
    return g0, g1, psi, pi, varlist


def solve_sw(params=None):
    g0, g1, psi, pi, varlist = build_sw_model(params)
    G1, C, impact, eu = gensys(g0, g1, np.zeros(g0.shape[0]), psi, pi)
    return dict(G1=G1, impact=impact, eu=eu, varlist=varlist,
                idx={v: i for i, v in enumerate(varlist)})


# --------------------------------------------------------------------------- #
# Data + Kalman-filter forecasting
# --------------------------------------------------------------------------- #
_SW_FRED = dict(gdp="GDPC1", cons="PCECC96", inv="GPDIC1", defl="GDPDEF",
                hours="HOANBS", wage="COMPRNFB", ffr="FEDFUNDS", pop="CNP16OV")


def load_sw_observables():
    """Fetch and transform the seven Smets–Wouters observables (quarterly) from FRED.

    Returns a DataFrame with columns [dy, dc, dinve, dw, labobs, pinfobs, robs],
    each demeaned (the model works in deviations). Raises if FRED is unavailable.
    """
    from . import ar  # noqa (ensure package import side effects are fine)
    from ..data import fred as _f

    if not _f.has_live_data():
        raise RuntimeError("Smets–Wouters needs live FRED data (set FRED_API_KEY).")

    start = "1965-01-01"
    raw = {k: _f._fetch_fred_series(sid, start) for k, sid in _SW_FRED.items()}
    if any(v is None for v in raw.values()):
        missing = [k for k, v in raw.items() if v is None]
        raise RuntimeError(f"FRED fetch failed for: {missing}")

    def q(s):  # to quarter-start index, averaging within quarter
        return s.resample("QS").mean()

    gdp, cons, inv = q(raw["gdp"]), q(raw["cons"]), q(raw["inv"])
    defl, hours, wage = q(raw["defl"]), q(raw["hours"]), q(raw["wage"])
    ffr, pop = q(raw["ffr"]), q(raw["pop"])

    df = pd.DataFrame({"gdp": gdp, "cons": cons, "inv": inv, "defl": defl,
                       "hours": hours, "wage": wage, "ffr": ffr, "pop": pop}).dropna()

    lgdp_pc = np.log(df["gdp"] / df["pop"])
    lc_pc = np.log(df["cons"] / df["pop"])
    li_pc = np.log(df["inv"] / df["pop"])
    lhours_pc = np.log(df["hours"] / df["pop"])
    lwage = np.log(df["wage"])
    lp = np.log(df["defl"])

    out = pd.DataFrame({
        "dy": 100 * lgdp_pc.diff(),
        "dc": 100 * lc_pc.diff(),
        "dinve": 100 * li_pc.diff(),
        "dw": 100 * lwage.diff(),
        "labobs": 100 * lhours_pc,
        "pinfobs": 100 * lp.diff(),
        "robs": df["ffr"] / 4.0,
    }).dropna()
    # store the (annualized) mean inflation before demeaning, for the forecast level
    out.attrs["pinf_mean_q"] = float(out["pinfobs"].mean())
    return out - out.mean()


OBS_ORDER = ["dy", "dc", "dinve", "dw", "labobs", "pinfobs", "robs"]


def _measurement(idx):
    """Selector rows for observables on the augmented state [y_t; y_{t-1}].
    Level observables use y_t only; growth observables use y_t - y_{t-1}."""
    n = len(idx)
    H = np.zeros((len(OBS_ORDER), 2 * n))
    level = {"labobs": "lab", "pinfobs": "pinf", "robs": "r"}
    growth = {"dy": "y", "dc": "c", "dinve": "inve", "dw": "w"}
    for i, ob in enumerate(OBS_ORDER):
        if ob in level:
            H[i, idx[level[ob]]] = 1.0
        else:
            v = growth[ob]
            H[i, idx[v]] = 1.0          # y_t
            H[i, n + idx[v]] = -1.0     # y_{t-1}
    return H


class SmetsWouters2007(ForecastModel):
    info = ModelInfo(
        key="sw07",
        name="Smets–Wouters (2007) DSGE (full, 7-shock)",
        family="Structural",
        reference="Smets & Wouters (2007), AER",
        description=(
            "The full medium-scale Smets–Wouters (2007) New Keynesian DSGE with seven "
            "structural shocks (productivity, risk premium, government spending, "
            "investment-specific, monetary, price mark-up, wage mark-up), capital with "
            "adjustment costs and variable utilization, habit formation, and Calvo "
            "price/wage setting with indexation. Deep parameters are set to the paper's "
            "estimated posterior mode; the model is solved with gensys and Kalman-"
            "filtered on the seven standard US observables from FRED to forecast "
            "inflation. Quarterly; forecasts GDP-deflator inflation."
        ),
        needs_activity=True,
        citation="Smets, F. & Wouters, R. (2007), 'Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach', American Economic Review 97(3).",
        intuition="A full quarterly model economy — households, firms, a central bank, capital accumulation — where seven identified shocks propagate through nominal and real frictions to move inflation.",
        unique="The most complete structural model here: it uses seven macro time series (output, consumption, investment, wages, hours, inflation, the policy rate) jointly, and decomposes inflation into its structural drivers.",
        strengths="The benchmark medium-scale DSGE for policy analysis; theory-consistent, identifies the shocks behind inflation, and its posterior-mode forecasts are a recognized structural benchmark.",
        caveats="Heavy (solves a ~45-variable RE system and filters 7 series); forecasts GDP-deflator inflation, so comparison against CPI/PCE is approximate. Parameters fixed at the 2007 posterior mode rather than re-estimated each period.",
        forecast_shape="A slow, theory-driven mean-reversion of inflation toward the model's steady state as the estimated shocks decay.",
    )

    # Names of SW07 parameters we let the UI override. Each corresponds to a
    # single deep parameter in SW_PARAMS.
    _USER_OVERRIDES = ("crpi", "crr", "chabb", "cprobp")

    def __init__(self, crpi: float | None = None, crr: float | None = None,
                 chabb: float | None = None, cprobp: float | None = None):
        # Build the params dict — user overrides win, else the SW07 posterior mode.
        overrides = dict(crpi=crpi, crr=crr, chabb=chabb, cprobp=cprobp)
        supplied = {k: float(v) for k, v in overrides.items() if v is not None}
        super().__init__(**supplied)
        self._param_overrides = supplied

    def _fit(self) -> None:
        # Merge SW07 posterior-mode params with any user overrides.
        params = dict(SW_PARAMS)
        params.update(self._param_overrides)
        sol = solve_sw(params)
        if sol["eu"] != [1, 1]:
            raise RuntimeError(
                f"SW07 has no unique stable solution (eu={sol['eu']}). "
                f"Try widening the Taylor-rule inflation response (crpi) or "
                f"lowering the habit / stickiness params."
            )
        self._sol = sol
        obs = load_sw_observables()
        self._pinf_mean_q = obs.attrs["pinf_mean_q"]
        Z = obs[OBS_ORDER].values

        G1, impact = sol["G1"], sol["impact"]
        n = G1.shape[0]
        sig2 = np.array([params[f"sig_{s}"] ** 2 for s in
                         ["a", "b", "g", "qs", "m", "pinf", "w"]])
        Q = impact @ np.diag(sig2) @ impact.T

        # augmented state [y_t; y_{t-1}]
        F = np.block([[G1, np.zeros((n, n))], [np.eye(n), np.zeros((n, n))]])
        Qa = np.block([[Q, np.zeros((n, n))], [np.zeros((n, n)), np.zeros((n, n))]])
        H = _measurement(sol["idx"])
        Rm = np.diag(0.05 * np.var(Z, axis=0) + 1e-6)

        # stationary initial covariance
        from scipy.linalg import solve_discrete_lyapunov
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = solve_discrete_lyapunov(G1, Q)
        P0 = np.block([[P, G1 @ P], [P @ G1.T, P]])
        s = np.zeros(2 * n)

        for t in range(Z.shape[0]):
            s = F @ s
            P0 = F @ P0 @ F.T + Qa
            yhat = H @ s
            v = Z[t] - yhat
            S = H @ P0 @ H.T + Rm
            K = P0 @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
            s = s + K @ v
            P0 = (np.eye(2 * n) - K @ H) @ P0

        self._state = s[:n]          # filtered y_T
        self._pinf_idx = sol["idx"]["pinf"]

    def _forecast(self, h: int) -> float:
        # dashboard passes h in periods of the selected frequency; SW is quarterly.
        hq = max(1, int(round(h / 3))) if getattr(self, "_freq", "M") == "M" else h
        G1 = self._sol["G1"]
        yh = np.linalg.matrix_power(G1, hq) @ self._state
        pinf_dev_q = float(yh[self._pinf_idx])          # quarterly deviation
        ann = 4.0 * (self._pinf_mean_q + pinf_dev_q)    # annualized level (%)
        return ann

    def fit(self, y, X=None):
        # remember frequency so the horizon can be mapped to quarters
        self._freq = "M" if (len(y) > 2 and (y.index[1] - y.index[0]).days < 45) else "Q"
        return super().fit(y, X)
