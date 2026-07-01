"""Bayesian VARs in [inflation, unemployment] with a Minnesota prior.

Two models:

* :class:`BVAR` — Minnesota (Litterman) prior with **fixed** overall shrinkage,
  implemented via Theil dummy observations, iterated forward for multi-step
  inflation forecasts.
* :class:`BVARHierarchical` — the Giannone–Lenza–Primiceri (2015) idea: instead of
  fixing the shrinkage hyperparameter λ, treat it as unknown and **choose it by
  maximizing the marginal likelihood** (empirical Bayes), with a Gamma hyperprior.
  This lets the data decide how tightly to shrink toward a random walk, which GLP show
  delivers forecasts competitive with factor models.

Both share the conjugate Normal–Inverse-Wishart machinery below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .base import ForecastModel, ModelInfo


# --------------------------------------------------------------------------- #
# Shared conjugate-VAR machinery
# --------------------------------------------------------------------------- #
def _ar1_resid_sd(x: np.ndarray) -> float:
    """Residual std of a univariate AR(1) — the GLP scaling for the prior."""
    x0, x1 = x[:-1], x[1:]
    b = np.dot(x0, x1) / np.dot(x0, x0) if np.dot(x0, x0) > 0 else 0.0
    e = x1 - b * x0
    return float(np.std(e, ddof=1)) or 1.0


def _build_regressors(Ymat: np.ndarray, p: int):
    """Stack VAR(p) regressors: X_t = [1, Y_{t-1}, ..., Y_{t-p}]."""
    T, k = Ymat.shape
    rows_y, rows_x = [], []
    for t in range(p, T):
        rows_y.append(Ymat[t])
        lags = [Ymat[t - l] for l in range(1, p + 1)]
        rows_x.append(np.concatenate([[1.0], *lags]))
    return np.asarray(rows_y), np.asarray(rows_x)


def _minnesota_dummies(k: int, p: int, sigmas: np.ndarray, lam: float):
    """Theil dummy observations encoding a Minnesota prior: own first lag ~ RW(1),
    all else ~ 0, tighter at longer lags; a loose prior on the intercept."""
    n_coef = 1 + k * p
    Yd, Xd = [], []
    for i in range(k):
        for l in range(1, p + 1):
            y_row = np.zeros(k)
            x_row = np.zeros(n_coef)
            prior_mean = 1.0 if l == 1 else 0.0
            tightness = (sigmas[i] * l) / lam
            y_row[i] = prior_mean * tightness
            x_row[1 + (l - 1) * k + i] = tightness
            Yd.append(y_row)
            Xd.append(x_row)
    # residual-scale dummies (prior on Sigma), one per variable
    for i in range(k):
        y_row = np.zeros(k)
        y_row[i] = sigmas[i]
        Yd.append(y_row)
        Xd.append(np.zeros(n_coef))
    # loose intercept prior
    y0 = np.zeros(k)
    x0 = np.zeros(n_coef)
    x0[0] = 1e-3
    Yd.append(y0)
    Xd.append(x0)
    return np.asarray(Yd), np.asarray(Xd)


def _log_multigamma(a: float, n: int) -> float:
    return (n * (n - 1) / 4.0) * np.log(np.pi) + sum(gammaln(a + (1 - j) / 2.0) for j in range(1, n + 1))


def _log_marginal_likelihood(Yr, Xr, Yd, Xd) -> float:
    """Log marginal likelihood of the actual data under the conjugate Minnesota prior
    represented by the dummy observations (Bańbura–Giannone–Reichlin 2010 /
    Giannone–Lenza–Primiceri 2015 closed form)."""
    n = Yr.shape[1]
    T = Yr.shape[0]
    k = Xr.shape[1]
    Td = Yd.shape[0]

    # prior moments (dummies only)
    XdXd = Xd.T @ Xd
    Bd = np.linalg.solve(XdXd + 1e-10 * np.eye(k), Xd.T @ Yd)
    Sd = Yd.T @ Yd - Bd.T @ XdXd @ Bd

    # posterior moments (actual + dummies)
    Xs = np.vstack([Xr, Xd])
    Ys = np.vstack([Yr, Yd])
    XsXs = Xs.T @ Xs
    Bs = np.linalg.solve(XsXs + 1e-10 * np.eye(k), Xs.T @ Ys)
    Ss = Ys.T @ Ys - Bs.T @ XsXs @ Bs

    dprior = Td - k          # prior degrees of freedom
    dpost = Td + T - k       # posterior degrees of freedom

    def _logdet(M):
        sign, ld = np.linalg.slogdet(M)
        return ld if sign > 0 else -1e12

    ll = (
        -(n * T / 2.0) * np.log(np.pi)
        + (n / 2.0) * (_logdet(XdXd) - _logdet(XsXs))
        + _log_multigamma(dpost / 2.0, n) - _log_multigamma(dprior / 2.0, n)
        + (dprior / 2.0) * _logdet(Sd) - (dpost / 2.0) * _logdet(Ss)
    )
    return float(ll)


class _BaseBVAR(ForecastModel):
    """Shared fit/forecast; subclasses supply the shrinkage λ via `_choose_lambda`."""

    def __init__(self, p: int = 4, activity_col: str = "unrate"):
        super().__init__(p=p, activity_col=activity_col)
        self.p = p
        self.activity_col = activity_col
        self.lam_ = None

    def _assemble(self):
        pi = self._y
        if self._X is not None and self.activity_col in getattr(self._X, "columns", []):
            act = self._X[self.activity_col].reindex(pi.index).ffill()
            Y = pd.concat([pi.rename("pi"), act.rename("act")], axis=1).dropna()
        else:
            Y = pi.rename("pi").to_frame().dropna()
        return Y

    def _fit(self) -> None:
        Y = self._assemble()
        Ymat = Y.values
        self._k = Ymat.shape[1]
        self._sigmas = np.array([_ar1_resid_sd(Ymat[:, i]) for i in range(self._k)])
        self._Yr, self._Xr = _build_regressors(Ymat, self.p)

        self.lam_ = self._choose_lambda()
        Yd, Xd = _minnesota_dummies(self._k, self.p, self._sigmas, self.lam_)
        Xs = np.vstack([self._Xr, Xd])
        Ys = np.vstack([self._Yr, Yd])
        self._B = np.linalg.solve(Xs.T @ Xs + 1e-10 * np.eye(Xs.shape[1]), Xs.T @ Ys)
        self._last = Ymat[-self.p:][::-1]  # most recent p obs, newest first

    def _choose_lambda(self) -> float:
        raise NotImplementedError

    def _forecast(self, h: int) -> float:
        p = self.p
        hist = list(self._last)
        pred = None
        for _ in range(h):
            x = np.concatenate([[1.0], *hist[:p]])
            yhat = x @ self._B
            hist.insert(0, yhat)
            pred = yhat
        return float(pred[0])


class BVAR(_BaseBVAR):
    info = ModelInfo(
        key="bvar",
        name="BVAR (Minnesota prior, fixed λ)",
        family="Statistical",
        reference="Litterman (1986); Bańbura–Giannone–Reichlin (2010)",
        description=(
            "A Bayesian vector autoregression in inflation and unemployment with a "
            "Minnesota prior that shrinks the dynamics toward independent random walks. "
            "The shrinkage strength λ is fixed. Captures joint inflation–slack dynamics "
            "while controlling overfitting."
        ),
        needs_activity=True,
        citation="Litterman, R. (1986), JBES; Bańbura, Giannone & Reichlin (2010), J. Applied Econometrics.",
        intuition="A two-variable system where inflation and unemployment predict each other and themselves, with a prior that pulls every equation gently toward 'no change'.",
        unique="The only multivariate model in the statistical group: it uses the joint dynamics of inflation and the labor market, not inflation alone.",
        strengths="Bayesian shrinkage tames the many VAR coefficients, so it forecasts better than an unrestricted VAR and exploits cross-variable feedback.",
        caveats="You must pick the shrinkage λ by hand; too loose overfits, too tight collapses to a random walk. The hierarchical version fixes this.",
        forecast_shape="A smooth mean-reverting path shaped by both inflation's and unemployment's recent moves.",
    )

    def __init__(self, p: int = 4, lam: float = 0.2, activity_col: str = "unrate"):
        super().__init__(p=p, activity_col=activity_col)
        self.lam = lam

    def _choose_lambda(self) -> float:
        return self.lam


class BVARHierarchical(_BaseBVAR):
    info = ModelInfo(
        key="bvarh",
        name="BVAR (hierarchical, GLP shrinkage)",
        family="Statistical",
        reference="Giannone–Lenza–Primiceri (2015)",
        description=(
            "The same inflation–unemployment BVAR, but the Minnesota shrinkage λ is "
            "treated as an unknown hyperparameter and chosen to maximize the marginal "
            "likelihood (with a Gamma hyperprior). This empirical-Bayes step lets the "
            "data set how tightly to shrink toward a random walk — the Giannone, Lenza & "
            "Primiceri (2015) approach, which is competitive with factor models."
        ),
        needs_activity=True,
        citation="Giannone, D., Lenza, M. & Primiceri, G. (2015), 'Prior Selection for Vector Autoregressions', REStat.",
        intuition="Same inflation–unemployment VAR as the fixed BVAR, but it automatically tunes how much to shrink by asking which λ best explains the historical data.",
        unique="Removes the arbitrary shrinkage choice: the level of Bayesian shrinkage is learned from the data via the marginal likelihood rather than set by the modeler.",
        strengths="Adapts shrinkage to the sample — tighter when data are uninformative, looser when they are rich. GLP show it rivals much larger models.",
        caveats="Still a linear VAR with constant coefficients; a structural break in the dynamics is not modeled. Optimizing λ adds a little compute.",
        forecast_shape="Like the fixed BVAR, but with a data-chosen amount of mean reversion — typically smoother when the sample is short.",
    )

    def __init__(self, p: int = 4, activity_col: str = "unrate",
                 lam_grid=None, gamma_prior=(0.5, 0.04)):
        super().__init__(p=p, activity_col=activity_col)
        # Gamma hyperprior on λ (mode ≈ 0.2, as in GLP); (shape scale) parameterization.
        self.lam_grid = np.asarray(lam_grid) if lam_grid is not None else np.linspace(0.02, 1.0, 25)
        self.gamma_prior = gamma_prior
        self.ml_curve_ = None

    def _log_hyperprior(self, lam: float) -> float:
        shape, scale = self.gamma_prior
        return (shape - 1) * np.log(lam) - lam / scale

    def _choose_lambda(self) -> float:
        scores = []
        for lam in self.lam_grid:
            Yd, Xd = _minnesota_dummies(self._k, self.p, self._sigmas, lam)
            try:
                ll = _log_marginal_likelihood(self._Yr, self._Xr, Yd, Xd) + self._log_hyperprior(lam)
            except np.linalg.LinAlgError:
                ll = -np.inf
            scores.append(ll)
        scores = np.asarray(scores)
        self.ml_curve_ = pd.Series(scores, index=self.lam_grid)
        return float(self.lam_grid[int(np.nanargmax(scores))])
