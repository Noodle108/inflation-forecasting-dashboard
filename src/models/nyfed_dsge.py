"""NY Fed–style DSGE with financial frictions.

The New York Fed's DSGE (Del Negro, Eusepi, Giannoni, Schorfheide and coauthors; the
"Liberty Street" model) is a Smets–Wouters style New Keynesian model *augmented with a
financial-friction block* — a Bernanke–Gertler–Gilchrist accelerator that lets credit
spreads drive real activity. It is the workhorse the NY Fed uses in its quarterly
inflation projections and has repeatedly been shown to fit the post-2008 disinflation
and the post-2020 surge better than the frictionless SW07 core, precisely because it
identifies a *financial* shock that shifts aggregate demand.

Rather than duplicate the full 7-shock SW07 implementation already in the dashboard
(`sw07`), this build isolates the NY Fed model's distinctive contribution — a
credit-spread wedge in the IS curve — inside a 4-shock New Keynesian core:

    x_t     = E_t x_{t+1} - (1/σ)(i_t - E_t π_{t+1}) + a_t - b_t     (IS)
    π_t     = β E_t π_{t+1} + κ x_t + u_t                            (NKPC)
    i_t     = φ_π π_t + φ_x x_t + v_t                                (Taylor rule)
    spread_t = θ b_t                                                 (observation)

with a_t (demand), u_t (cost-push), v_t (monetary) and b_t (financial-friction wedge)
each AR(1). Because the states are four independent AR(1)s, the model's
minimum-state-variable solution is a 2×4 loading matrix G solved shock-by-shock —
which is numerically bullet-proof and needs no QZ decomposition. Shock persistences,
variances, and the spread-loading θ are estimated by Kalman-filter maximum likelihood
on three observables (inflation, output/labour gap, BAA–10Y credit spread).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo


def _solve_nk_fin(beta, sigma, kappa, phi_pi, phi_x, rhos):
    """MSV solution of the 4-shock NK model. Returns G (2 x 4) where the columns are
    [a (demand), u (cost-push), v (monetary), b (financial wedge)]. b enters the IS
    curve with the opposite sign of a, so its column is exactly the negative of a's.
    """
    G = np.zeros((2, 4))
    for j, r in enumerate(rhos):
        is_a, is_u, is_v, is_b = (j == 0), (j == 1), (j == 2), (j == 3)
        M = np.array([
            [r - 1.0 - phi_x / sigma, (r - phi_pi) / sigma],
            [kappa,                    beta * r - 1.0],
        ])
        rhs = np.array([
            (1.0 / sigma) * (1.0 if is_v else 0.0)
            - (1.0 if is_a else 0.0)
            + (1.0 if is_b else 0.0),
            -(1.0 if is_u else 0.0),
        ])
        x_pi = np.linalg.solve(M, rhs)
        G[0, j] = x_pi[0]
        G[1, j] = x_pi[1]
    return G


def _kalman_ll_fin(theta, Z, deep, Rm):
    """Kalman log-likelihood.
    theta = [rho_a,rho_u,rho_v,rho_b, q_a,q_u,q_v,q_b, theta_spread].
    Observables Z columns: [pi (inflation gap), x (activity gap), s (spread gap)].
    """
    beta, sigma, kappa, phi_pi, phi_x = deep
    rhos = np.clip(theta[:4], 0.0, 0.995)
    q = np.clip(theta[4:8], 1e-6, None)
    theta_s = float(theta[8])
    try:
        G = _solve_nk_fin(beta, sigma, kappa, phi_pi, phi_x, rhos)
    except np.linalg.LinAlgError:
        return -1e12, np.zeros(4)

    R = np.diag(rhos)
    Q = np.diag(q)
    # H rows: inflation = G[1,:], gap = G[0,:], spread = theta_s * e_b
    H = np.zeros((3, 4))
    H[0, :] = G[1, :]
    H[1, :] = G[0, :]
    H[2, 3] = theta_s  # spread loads only on the financial shock

    s = np.zeros(4)
    P = np.diag(q / np.maximum(1.0 - rhos ** 2, 1e-3))
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
        ll += -0.5 * (3 * np.log(2 * np.pi) + ld + v @ Sinv @ v)
        K = P @ H.T @ Sinv
        s = s + K @ v
        P = (np.eye(4) - K @ H) @ P
    return float(ll), s


def _load_spread():
    """Monthly BAA–10Y credit spread from FRED (percent, annualized)."""
    from ..data import fred as _f
    if not _f.has_live_data():
        return None
    s = _f._fetch_fred_series("BAA10YM", "1960-01-01")
    return s


class NYFedDSGE(ForecastModel):
    info = ModelInfo(
        key="nyfed",
        name="NY Fed DSGE (financial-friction NK)",
        family="Structural",
        reference="Del Negro–Giannoni–Schorfheide (2015); NY Fed DSGE",
        description=(
            "A New Keynesian DSGE augmented with a Bernanke–Gertler–Gilchrist credit "
            "wedge: a financial-friction shock enters the IS curve like an inverse "
            "demand shock and is identified from the BAA–10Y credit spread. The model "
            "is solved for its rational-expectations equilibrium and its four shock "
            "processes (demand, cost-push, monetary, financial) are estimated by "
            "maximum likelihood via the Kalman filter on three observables. Isolates "
            "the NY Fed DSGE's distinctive contribution — financial frictions — inside "
            "a transparent, digestible core."
        ),
        needs_activity=True,
        citation="Del Negro, M., Giannoni, M. & Schorfheide, F. (2015), 'Inflation in the Great Recession and New Keynesian Models', AEJ:Macro; NY Fed Liberty Street DSGE projections.",
        intuition="Adds Wall Street to Main Street's NK model: when credit spreads widen, borrowing costs go up, spending falls, and inflation cools — even without a policy move.",
        unique="The only model here that uses credit-market data (BAA–10Y spread) to identify a financial shock; separates spread-driven from monetary and demand disinflation.",
        strengths="Captures the 2008–09 disinflation and the post-2020 credit-driven episodes better than a frictionless DSGE; interpretable in terms of financial vs. real shocks.",
        caveats="A stylized 4-shock version of the NY Fed's full model (which includes wages, capital, capacity utilization). Deep parameters calibrated, not estimated.",
        forecast_shape="A mean-reverting path whose speed depends on which shock currently dominates — persistent credit spreads drag inflation down for longer than monetary shocks.",
    )

    DEEP_DEFAULT = dict(beta=0.99, sigma=1.0, kappa=0.05, phi_pi=1.5, phi_x=0.125)

    def __init__(self, activity_col: str = "ngap", okun: float = -2.0,
                 sigma: float | None = None, kappa: float | None = None,
                 phi_pi: float | None = None, phi_x: float | None = None):
        super().__init__(activity_col=activity_col, okun=okun,
                         sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x)
        self.activity_col = activity_col
        self.okun = okun
        # Same 4-knob deep-param override as SmallScaleDSGE.
        self.DEEP = dict(self.DEEP_DEFAULT)
        if sigma is not None:  self.DEEP["sigma"] = float(sigma)
        if kappa is not None:  self.DEEP["kappa"] = float(kappa)
        if phi_pi is not None: self.DEEP["phi_pi"] = float(phi_pi)
        if phi_x is not None:  self.DEEP["phi_x"] = float(phi_x)

    def _fit(self) -> None:
        from scipy.optimize import minimize

        pi = self._y
        self._pi_mean = float(pi.mean())
        pit = (pi - self._pi_mean).values

        # Activity gap (from unemployment gap via Okun's law), demeaned.
        if self._X is not None and self.activity_col in getattr(self._X, "columns", []):
            ugap = self._X[self.activity_col].reindex(pi.index).ffill()
            gap = (self.okun * ugap).values
            gap = gap - np.nanmean(gap)
        else:
            gap = np.zeros_like(pit)

        # Credit spread, aligned to pi's index (monthly or quarterly), demeaned.
        spread_raw = _load_spread()
        if spread_raw is None:
            # Fall back to the frictionless 3-shock model by feeding a zero spread.
            spread = np.zeros_like(pit)
            self._spread_mean = 0.0
        else:
            if pi.index.freqstr and pi.index.freqstr.startswith("Q"):
                spread_raw = spread_raw.resample("QS").mean()
            spread_s = spread_raw.reindex(pi.index).ffill()
            self._spread_mean = float(spread_s.mean())
            spread = (spread_s - self._spread_mean).values

        Z = np.column_stack([pit, gap, spread])
        Z = Z[~np.isnan(Z).any(axis=1)]

        deep = tuple(self.DEEP[k] for k in ["beta", "sigma", "kappa", "phi_pi", "phi_x"])
        # 10% measurement-error variance per observable
        Rm = np.diag(0.1 * np.var(Z, axis=0) + 1e-6)

        def neg_ll(theta):
            ll, _ = _kalman_ll_fin(theta, Z, deep, Rm)
            return -ll

        x0 = np.array([0.7, 0.6, 0.3, 0.9,  0.4, 0.4, 0.3, 0.2,  1.0])
        bounds = [(0.0, 0.97)] * 4 + [(1e-4, 25.0)] * 4 + [(0.01, 5.0)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 120})
            theta = res.x
        except Exception:
            theta = x0

        self.theta_ = theta
        self._rhos = np.clip(theta[:4], 0.0, 0.995)
        _, s_final = _kalman_ll_fin(theta, Z, deep, Rm)
        self._state = s_final
        G = _solve_nk_fin(*deep, self._rhos)
        self._g_pi = G[1, :]

    def _forecast(self, h: int) -> float:
        decayed = (self._rhos ** h) * self._state
        pit = float(self._g_pi @ decayed)
        return pit + self._pi_mean
