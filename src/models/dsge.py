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
        reference="Galí–Gertler (1999); Bernanke–Blanchard (2023) anchoring",
        description=(
            "A hybrid New Keynesian Phillips curve estimated in **expectation-gap form**: "
            "(π − π^LR) is regressed on its own lag and the output gap, so that in the "
            "absence of shocks inflation converges to long-run expected inflation π^LR "
            "(Cleveland Fed EXPINF10YR when available; sample mean as fallback). Follows "
            "Bernanke–Blanchard (2023) in imposing the long-run anchor explicitly rather "
            "than letting the unconditional mean of the fitted regression set it."
        ),
        needs_activity=True,
        citation="Galí, J. & Gertler, M. (1999), 'Inflation Dynamics: A Structural Econometric Analysis', JME.",
        intuition="Firms set prices partly on where they expect inflation to go (anchor) and partly on how hot the economy is (the output gap). Written so the anchor is π^LR from surveys.",
        unique="A single structural equation linking inflation to the output gap, with a hard long-run anchor — the theoretical backbone the full DSGE embeds.",
        strengths="Interpretable and cheap; makes the inflation–gap trade-off explicit; forecast converges to survey-based expectations rather than to the sample mean.",
        caveats="Not a complete equilibrium model: expectations and the gap are still reduced-form. The κ slope is famously flat in post-1985 data.",
        forecast_shape="A mean-reverting path pulled toward π^LR (currently ~2.5% from Cleveland Fed) at a speed set by the persistence coefficient b.",
    )

    def __init__(self, okun: float = -2.0, activity_col: str = "ngap",
                 anchor_col: str = "exp10yr", anchor_override: float | None = None):
        super().__init__(okun=okun, activity_col=activity_col,
                         anchor_col=anchor_col,
                         anchor_override=anchor_override)
        self.okun = okun
        self.activity_col = activity_col
        self.anchor_col = anchor_col
        # If set, replaces the EXPINF10YR-derived anchor with a constant value
        # (in annualized-percent units). Lets the UI ask "what if the long-run
        # anchor were 2% instead of 2.5%?".
        self.anchor_override = anchor_override

    def _fit(self) -> None:
        import statsmodels.api as sm

        pi = self._y
        Xdf = self._X if self._X is not None else pd.DataFrame(index=pi.index)

        # Long-run anchor π^LR: user override > EXPINF10YR > sample mean.
        if self.anchor_override is not None:
            anchor = pd.Series(float(self.anchor_override), index=pi.index)
        elif self.anchor_col in getattr(Xdf, "columns", []):
            anchor = Xdf[self.anchor_col].reindex(pi.index).ffill().bfill()
        else:
            anchor = pd.Series(float(pi.mean()), index=pi.index)
        self._anchor = anchor
        self._last_anchor = float(anchor.iloc[-1])

        # Output gap (from unemployment gap via Okun's law).
        if self.activity_col in getattr(Xdf, "columns", []):
            ugap = Xdf[self.activity_col].reindex(pi.index).ffill()
        else:
            ugap = pd.Series(0.0, index=pi.index)
        gap = self.okun * ugap

        # Expectation-gap form: (π - π^LR) = b (π_{t-1} - π^LR) + κ · gap + ε.
        # **No intercept** — this forces the long-run destination to be π^LR
        # itself. Adding a constant would let OLS fit c = mean(π_gap) - b·..., and
        # the forecast steady state c/(1-b) would recover the sample mean of
        # π regardless of the anchor override. Omitting c makes the anchor
        # actually bind.
        df = pd.DataFrame({
            "pi_gap": pi - anchor,
            "pi_gap_l1": (pi - anchor).shift(1),
            "gap": gap,
            "gap_l1": gap.shift(1),
        }).dropna()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xp = df[["pi_gap_l1", "gap"]].values          # no add_constant
            self._nkpc = sm.OLS(df["pi_gap"].values, Xp).fit()
            # The gap AR still gets an intercept (small-sample stability).
            Xg = sm.add_constant(df["gap_l1"].values)
            self._gap_ar = sm.OLS(df["gap"].values, Xg).fit()

        self._last_pi_gap = float(df["pi_gap"].iloc[-1])
        self._last_gap = float(df["gap"].iloc[-1])

    def _forecast(self, h: int) -> float:
        b_pi, k_pi = self._nkpc.params
        c_g, rho_g = self._gap_ar.params
        pi_gap, gap = self._last_pi_gap, self._last_gap
        for _ in range(h):
            gap = c_g + rho_g * gap
            pi_gap = b_pi * pi_gap + k_pi * gap
        return float(self._last_anchor + pi_gap)

    def steady_state(self) -> float:
        """Long-run forecast the model converges to.

        With no intercept in the pi_gap regression, in a shock-free world
        pi_gap → κ·gap_ss / (1 - b), so long-run inflation = anchor + that.
        """
        b_pi, k_pi = self._nkpc.params
        c_g, rho_g = self._gap_ar.params
        if abs(rho_g) >= 1 or abs(b_pi) >= 1:
            return float(self._last_anchor)
        gap_ss = c_g / (1 - rho_g)
        pi_gap_ss = (k_pi * gap_ss) / (1 - b_pi)
        return float(self._last_anchor + pi_gap_ss)


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

    # calibrated deep parameters (standard textbook values). Defaults follow
    # Galí (2008); the four non-β params are user-tunable through __init__.
    DEEP_DEFAULT = dict(beta=0.99, sigma=1.0, kappa=0.05, phi_pi=1.5, phi_x=0.125)

    def __init__(self, activity_col: str = "ngap", okun: float = -2.0,
                 sigma: float | None = None, kappa: float | None = None,
                 phi_pi: float | None = None, phi_x: float | None = None):
        super().__init__(activity_col=activity_col, okun=okun,
                         sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x)
        self.activity_col = activity_col
        self.okun = okun
        # Build the calibration dict — user overrides win, else textbook default.
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
