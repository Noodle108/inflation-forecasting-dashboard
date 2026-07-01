"""Time-varying-parameter VAR with stochastic volatility (Primiceri 2005).

A VAR in [inflation, unemployment] whose coefficients **drift over time** and whose
shock variances follow **stochastic volatility** — the workhorse for capturing the
changing dynamics and changing volatility of postwar US inflation (Cogley–Sargent 2005;
Primiceri 2005). Estimated by Gibbs sampling:

* the time-varying coefficients of each equation are drawn with a Carter–Kohn
  forward-filter-backward-sample step (a random-walk law of motion for the coefficients);
* each equation's log-variance is drawn with the same single-move stochastic-volatility
  sampler used by the UCSV-SV model.

The forecast iterates the VAR forward from the *end-of-sample* coefficients (the relevant
ones for the future under random-walk drift). This build uses independent equations with
time-varying diagonal volatility (a common simplification of Primiceri's triangular
reduction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo
from .ucsv import _sv_single_move


def _carter_kohn(y, X, Q, h, b0, P0, rng):
    """Draw the time-varying coefficient path for y_t = x_t' beta_t + eps_t, with
    beta_t = beta_{t-1} + u_t (Var u = Q) and Var(eps_t) = exp(h_t). Returns the
    sampled beta path (T x m)."""
    T, m = X.shape
    a = np.zeros((T, m)); R = np.zeros((T, m, m))
    mm = np.zeros((T, m)); CC = np.zeros((T, m, m))
    prev_m, prev_C = b0, P0
    for t in range(T):
        a[t] = prev_m
        R[t] = prev_C + Q
        x = X[t]
        S = x @ R[t] @ x + np.exp(h[t])
        K = R[t] @ x / S
        mm[t] = a[t] + K * (y[t] - x @ a[t])
        CC[t] = R[t] - np.outer(K, x @ R[t])
        prev_m, prev_C = mm[t], CC[t]

    beta = np.zeros((T, m))
    CC[-1] = 0.5 * (CC[-1] + CC[-1].T) + 1e-10 * np.eye(m)
    beta[-1] = rng.multivariate_normal(mm[-1], CC[-1])
    for t in range(T - 2, -1, -1):
        Rn = R[t + 1]
        Rn_inv = np.linalg.inv(Rn + 1e-10 * np.eye(m))
        C_t = CC[t]
        gain = C_t @ Rn_inv
        m_bar = mm[t] + gain @ (beta[t + 1] - a[t + 1])
        C_bar = C_t - gain @ Rn @ gain.T
        C_bar = 0.5 * (C_bar + C_bar.T) + 1e-10 * np.eye(m)
        beta[t] = rng.multivariate_normal(m_bar, C_bar)
    return beta


class TVPVARSV(ForecastModel):
    info = ModelInfo(
        key="tvpvar",
        name="TVP-VAR with stochastic volatility",
        family="Statistical",
        reference="Primiceri (2005); Cogley–Sargent (2005)",
        description=(
            "A vector autoregression in inflation and unemployment whose coefficients "
            "drift over time and whose shock volatilities evolve as stochastic "
            "volatility, estimated by Gibbs sampling (Carter–Kohn for the coefficients, "
            "a single-move sampler for the volatilities). Captures the changing "
            "persistence and changing volatility of inflation that constant-parameter "
            "models miss. Forecasts from the end-of-sample coefficients."
        ),
        needs_activity=True,
        citation="Primiceri, G. (2005), 'Time Varying Structural VARs and Monetary Policy', Review of Economic Studies 72(3); Cogley & Sargent (2005), RED.",
        intuition="A VAR whose relationships are allowed to slowly change every quarter, and whose surprise-sizes grow and shrink over time — so it adapts as the economy's behavior shifts.",
        unique="The only model here with both drifting coefficients and changing volatility; it lets the inflation–unemployment relationship itself evolve rather than assuming it is fixed.",
        strengths="Tracks structural change (the Great Inflation, the Great Moderation, post-2020) without a hard break; evidence shows the stochastic-volatility part especially improves forecasts.",
        caveats="Heavy: runs an MCMC chain per fit. This build uses diagonal time-varying volatility rather than Primiceri's full triangular reduction.",
        forecast_shape="A VAR path shaped by the most recent (drifted) dynamics of inflation and unemployment.",
    )

    def __init__(self, p: int = 2, n_draws: int = 150, burn: int = 150,
                 kappa: float = 0.02, gamma: float = 0.2, activity_col: str = "unrate",
                 seed: int = 20):
        super().__init__(p=p, n_draws=n_draws, burn=burn, kappa=kappa, activity_col=activity_col)
        self.p = p
        self.n_draws = n_draws
        self.burn = burn
        self.kappa2 = kappa ** 2
        self.gamma2 = gamma ** 2
        self.activity_col = activity_col
        self.seed = seed

    def _fit(self) -> None:
        pi = self._y
        if self._X is not None and self.activity_col in getattr(self._X, "columns", []):
            act = self._X[self.activity_col].reindex(pi.index).ffill()
            Y = pd.concat([pi.rename("pi"), act.rename("act")], axis=1).dropna()
        else:
            Y = pi.rename("pi").to_frame().dropna()
        self._names = list(Y.columns)
        Ymat = Y.values
        T, k = Ymat.shape
        self._k = k
        p = self.p

        # regressors x_t = [1, Y_{t-1}, ..., Y_{t-p}]
        rows_y, rows_x = [], []
        for t in range(p, T):
            rows_y.append(Ymat[t])
            rows_x.append(np.concatenate([[1.0], *[Ymat[t - l] for l in range(1, p + 1)]]))
        Yr = np.asarray(rows_y)             # Teff x k
        Xr = np.asarray(rows_x)             # Teff x m
        Teff, m = Xr.shape
        rng = np.random.default_rng(self.seed)

        beta_T_sum = np.zeros((k, m))
        kept = 0
        for eq in range(k):
            y_eq = Yr[:, eq]
            # OLS prior from the first block of data
            n0 = min(max(4 * m, 40), Teff // 2)
            X0, y0 = Xr[:n0], y_eq[:n0]
            XtX = X0.T @ X0 + 1e-6 * np.eye(m)
            b0 = np.linalg.solve(XtX, X0.T @ y0)
            resid0 = y0 - X0 @ b0
            s2 = float(resid0 @ resid0) / max(n0 - m, 1)
            V = s2 * np.linalg.inv(XtX)
            P0 = 4.0 * V
            Q = self.kappa2 * V

            # init volatility path from OLS residuals over full sample
            r_full = y_eq - Xr @ b0
            h = np.log(np.maximum(pd.Series(r_full ** 2).rolling(9, min_periods=1).mean().values, 1e-4))
            beta = np.tile(b0, (Teff, 1))

            for it in range(self.burn + self.n_draws):
                beta = _carter_kohn(y_eq, Xr, Q, h, b0, P0, rng)
                resid = y_eq - np.einsum("tm,tm->t", Xr, beta)
                _sv_single_move(resid, h, self.gamma2, rng)
                if it >= self.burn:
                    beta_T_sum[eq] += beta[-1]
                    if eq == 0:
                        kept += 1
            # kept counts draws for eq 0; same for all eqs
        self._B_T = beta_T_sum / self.n_draws     # k x m, end-of-sample coefficients
        self._last = Ymat[-p:][::-1]               # newest first
        self._m = m

    def _forecast(self, h: int) -> float:
        p, k = self.p, self._k
        hist = list(self._last)
        pred = None
        for _ in range(h):
            x = np.concatenate([[1.0], *hist[:p]])
            yhat = self._B_T @ x
            hist.insert(0, yhat)
            pred = yhat
        return float(pred[0])
