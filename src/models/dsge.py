"""Structural models.

Two structural forecasters of increasing fidelity:

* :class:`NewKeynesianPC` — a semi-structural hybrid New Keynesian Phillips curve
  (Galí–Gertler 1999): inflation driven by lagged inflation and the output gap, with
  the gap following an estimated AR process. Cheap and transparent.

* :class:`SmallScaleDSGE` — a genuine **micro-founded, estimated** three-equation
  New Keynesian DSGE (dynamic IS curve, NK Phillips curve, Taylor rule) in the
  Smets–Wouters (2007) tradition. Deep parameters are calibrated to standard values;
  the three structural shock processes are **estimated by maximum likelihood** with a
  Kalman filter, and the model is solved exactly for its rational-expectations
  equilibrium before being projected forward.

This is a *small-scale* DSGE, not the full 7-shock Smets–Wouters system (which needs a
dedicated solver like Dynare). That is a deliberate choice: the dashboard's goal is a
transparent, digestible structural forecast that can be compared like-for-like against
the statistical models, not a black box.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo


# =========================================================================== #
# 1) Semi-structural hybrid New Keynesian Phillips curve
# =========================================================================== #
class NewKeynesianPC(ForecastModel):
    info = ModelInfo(
        key="nkpc",
        name="New Keynesian Phillips Curve (semi-structural)",
        family="Structural",
        reference="Galí–Gertler (1999)",
        description=(
            "A hybrid New Keynesian Phillips curve: inflation is driven by lagged "
            "inflation and the output gap (from the unemployment gap via Okun's law), "
            "with the gap following an estimated AR process so the system can be "
            "iterated forward. A transparent, lightweight structural forecast."
        ),
        needs_activity=True,
        citation="Galí, J. & Gertler, M. (1999), 'Inflation Dynamics: A Structural Econometric Analysis', JME.",
        intuition="Firms set prices partly on where they expect inflation to go and partly on how hot the economy is (the output gap).",
        unique="A single structural equation linking inflation to the output gap — the theoretical backbone that the full DSGE embeds inside a complete model.",
        strengths="Interpretable and cheap; makes the inflation–gap trade-off explicit without a full general-equilibrium solve.",
        caveats="Not a complete equilibrium model: expectations and the gap are handled by reduced-form proxies rather than solved jointly.",
        forecast_shape="A mean-reverting path pulled by the estimated output gap and inflation persistence.",
    )

    def __init__(self, okun: float = -2.0, activity_col: str = "ngap"):
        super().__init__(okun=okun, activity_col=activity_col)
        self.okun = okun
        self.activity_col = activity_col

    def _fit(self) -> None:
        import statsmodels.api as sm

        pi = self._y
        if self._X is not None and self.activity_col in getattr(self._X, "columns", []):
            ugap = self._X[self.activity_col].reindex(pi.index).ffill()
        else:
            ugap = pd.Series(0.0, index=pi.index)
        gap = self.okun * ugap

        df = pd.DataFrame({"pi": pi, "gap": gap}).dropna()
        df["pi_l1"] = df["pi"].shift(1)
        df["gap_l1"] = df["gap"].shift(1)
        d = df.dropna()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xp = sm.add_constant(d[["pi_l1", "gap"]].values)
            self._nkpc = sm.OLS(d["pi"].values, Xp).fit()
            Xg = sm.add_constant(d["gap_l1"].values)
            self._gap_ar = sm.OLS(d["gap"].values, Xg).fit()

        self._last_pi = float(d["pi"].iloc[-1])
        self._last_gap = float(d["gap"].iloc[-1])

    def _forecast(self, h: int) -> float:
        c_pi, b_pi, k_pi = self._nkpc.params
        c_g, rho_g = self._gap_ar.params
        pi, gap = self._last_pi, self._last_gap
        for _ in range(h):
            gap = c_g + rho_g * gap
            pi = c_pi + b_pi * pi + k_pi * gap
        return float(pi)


# =========================================================================== #
# 2) Small-scale estimated New Keynesian DSGE
# =========================================================================== #
def _solve_nk(beta, sigma, kappa, phi_pi, phi_x, rhos):
    """Minimum-state-variable solution of the 3-equation NK model.

    Because the only states are three AR(1) shocks (demand a, cost-push u, monetary v),
    the jump variables (output gap x, inflation pi) are exact linear functions of the
    shocks: [x; pi] = G @ [a; u; v]. Solving reduces to one 2x2 system per shock, which
    is numerically bullet-proof and needs no QZ decomposition. Returns G (2x3).
    """
    G = np.zeros((2, 3))
    for j, r in enumerate(rhos):
        is_a, is_u, is_v = (j == 0), (j == 1), (j == 2)
        M = np.array([
            [r - 1.0 - phi_x / sigma, (r - phi_pi) / sigma],
            [kappa,                    beta * r - 1.0],
        ])
        rhs = np.array([
            (1.0 / sigma) * (1.0 if is_v else 0.0) - (1.0 if is_a else 0.0),
            -(1.0 if is_u else 0.0),
        ])
        x_pi = np.linalg.solve(M, rhs)   # [X (gap), Pi (inflation)]
        G[0, j] = x_pi[0]
        G[1, j] = x_pi[1]
    return G


def _kalman_ll(theta, Z, deep, Rm):
    """Kalman-filter log-likelihood of observables Z (T x m) given shock params
    theta = [rho_a, rho_u, rho_v, q_a, q_u, q_v]. Returns (loglik, final_state)."""
    beta, sigma, kappa, phi_pi, phi_x = deep
    rhos = np.clip(theta[:3], 0.0, 0.995)
    q = np.clip(theta[3:], 1e-6, None)
    try:
        G = _solve_nk(beta, sigma, kappa, phi_pi, phi_x, rhos)
    except np.linalg.LinAlgError:
        return -1e12, np.zeros(3)

    R = np.diag(rhos)
    Q = np.diag(q)
    m = Z.shape[1]
    # observation matrix: row 0 = inflation (G row 1), row 1 = gap (G row 0)
    if m == 2:
        H = np.vstack([G[1, :], G[0, :]])
    else:
        H = G[1, :].reshape(1, 3)
    Rm = np.atleast_2d(Rm)[:m, :m]

    s = np.zeros(3)
    P = np.diag(q / np.maximum(1.0 - rhos ** 2, 1e-3))  # stationary init
    ll = 0.0
    for t in range(Z.shape[0]):
        s = R @ s
        P = R @ P @ R.T + Q
        yhat = H @ s
        v = Z[t] - yhat
        S = H @ P @ H.T + Rm
        try:
            Sinv = np.linalg.inv(S)
            sign, ld = np.linalg.slogdet(S)
        except np.linalg.LinAlgError:
            return -1e12, s
        if sign <= 0:
            return -1e12, s
        ll += -0.5 * (m * np.log(2 * np.pi) + ld + v @ Sinv @ v)
        K = P @ H.T @ Sinv
        s = s + K @ v
        P = (np.eye(3) - K @ H) @ P
    return float(ll), s


class SmallScaleDSGE(ForecastModel):
    info = ModelInfo(
        key="dsge",
        name="New Keynesian DSGE (small-scale, estimated)",
        family="Structural",
        reference="Smets–Wouters (2007) tradition; small-scale NK model",
        description=(
            "A micro-founded three-equation New Keynesian DSGE: a dynamic IS curve, a "
            "New Keynesian Phillips curve, and a Taylor rule, driven by demand, "
            "cost-push, and monetary shocks. Deep parameters are calibrated to standard "
            "values; the shock persistences and variances are estimated by maximum "
            "likelihood via the Kalman filter. The model is solved for its "
            "rational-expectations equilibrium and projected forward. A digestible "
            "stand-in for the full Smets–Wouters DSGE."
        ),
        needs_activity=True,
        citation="Smets, F. & Wouters, R. (2007), 'Shocks and Frictions in US Business Cycles', AER (full DSGE); small-scale NK core à la Galí (2008).",
        intuition="A complete mini-economy: households choose spending, firms set prices, and the central bank sets rates by rule — inflation emerges from all three interacting.",
        unique="The only model here derived from optimizing households and firms: forecasts come from economic structure and identified shocks, not fitted correlations.",
        strengths="Imposes theory-consistent restrictions, so it degrades gracefully out of sample and decomposes inflation into demand vs. cost-push vs. policy shocks.",
        caveats="A stylized 3-equation core with calibrated deep parameters — not the full 7-shock Smets–Wouters; misspecification can bias forecasts if the structure is wrong.",
        forecast_shape="A path that mean-reverts to the inflation target as the estimated shocks decay, at speeds set by their persistence.",
    )

    # calibrated deep parameters (standard textbook values)
    DEEP = dict(beta=0.99, sigma=1.0, kappa=0.05, phi_pi=1.5, phi_x=0.125)

    def __init__(self, activity_col: str = "ngap", okun: float = -2.0):
        super().__init__(activity_col=activity_col, okun=okun)
        self.activity_col = activity_col
        self.okun = okun

    def _fit(self) -> None:
        from scipy.optimize import minimize

        pi = self._y
        self._pi_mean = float(pi.mean())
        pit = (pi - self._pi_mean).values

        if self._X is not None and self.activity_col in getattr(self._X, "columns", []):
            ugap = self._X[self.activity_col].reindex(pi.index).ffill()
            gap = (self.okun * ugap).values
            gap = gap - np.nanmean(gap)
            Z = np.column_stack([pit, gap])
        else:
            Z = pit.reshape(-1, 1)
        Z = Z[~np.isnan(Z).any(axis=1)]

        deep = tuple(self.DEEP[k] for k in ["beta", "sigma", "kappa", "phi_pi", "phi_x"])
        # fixed measurement-error variances (allow model misfit): 10% of obs variance
        Rm = np.diag(0.1 * np.var(Z, axis=0) + 1e-6)

        def neg_ll(theta):
            ll, _ = _kalman_ll(theta, Z, deep, Rm)
            return -ll

        x0 = np.array([0.6, 0.6, 0.3, 0.5, 0.5, 0.3])
        bounds = [(0.0, 0.97)] * 3 + [(1e-4, 25.0)] * 3
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 80})
            theta = res.x
        except Exception:
            theta = x0

        self.theta_ = theta
        self._rhos = np.clip(theta[:3], 0.0, 0.995)
        _, s_final = _kalman_ll(theta, Z, deep, Rm)
        self._state = s_final
        G = _solve_nk(*deep, self._rhos)
        self._g_pi = G[1, :]   # inflation loadings on shocks

    def _forecast(self, h: int) -> float:
        # E_t[shocks_{t+h}] = R^h s_t  (R diagonal) → inflation = g_pi · that
        decayed = (self._rhos ** h) * self._state
        pit = float(self._g_pi @ decayed)
        return pit + self._pi_mean
