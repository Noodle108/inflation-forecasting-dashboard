"""Unobserved-components / stochastic-volatility trend inflation.

Stock–Watson (2007) model inflation as trend + noise where the trend follows a random
walk and both innovation variances drift (stochastic volatility):

    pi_t   = tau_t + eps_t,     eps_t ~ N(0, sigma_eps_t^2)
    tau_t  = tau_{t-1} + eta_t, eta_t ~ N(0, sigma_eta_t^2)
    log sigma_eps_t^2 = log sigma_eps_{t-1}^2 + nu_eps_t,   nu_eps ~ N(0, gamma^2)
    log sigma_eta_t^2 = log sigma_eta_{t-1}^2 + nu_eta_t,   nu_eta ~ N(0, gamma^2)

The h-step forecast is simply the current trend, `pi_{t+h|t} = tau_T`.

Two implementations are provided:

* :class:`UCSV` — the trend estimated with the Kalman filter on a *constant-variance*
  local-level model (fast; a good approximation to the trend point forecast).
* :class:`UCSVSV` — the **full stochastic-volatility** model estimated by MCMC
  (Gibbs sampling): a forward-filter-backward-sample draw of the trend given the two
  volatility paths, and a Jacquier–Polson–Rossi single-move sampler for each
  log-volatility path given the trend. This is the canonical Stock–Watson (2007) object.
"""
from __future__ import annotations

import warnings

import numpy as np

from .base import ForecastModel, ModelInfo


class UCSV(ForecastModel):
    info = ModelInfo(
        key="ucsv",
        name="UCSV trend (local level)",
        family="Statistical",
        reference="Stock–Watson (2007)",
        description=(
            "Decomposes inflation into a slow-moving stochastic trend plus transitory "
            "noise, estimated with a constant-variance Kalman filter. The forecast at "
            "any horizon is the current trend. Fast approximation to the full UCSV; use "
            "the 'UCSV-SV' model for the stochastic-volatility version."
        ),
        citation="Stock, J. & Watson, M. (2007), 'Why Has US Inflation Become Harder to Forecast?', JMCB (constant-variance version).",
        intuition="Splits inflation into a slow-moving 'trend' and transitory noise, then forecasts the trend and ignores the noise.",
        unique="Forecasts an estimate of *underlying* inflation rather than the last raw print — a smarter random walk.",
        strengths="Filters out one-off shocks, so it captures where inflation is really centered; a strong modern statistical benchmark.",
        caveats="Assumes fixed noise/trend variances; can't tell whether a move is trend or noise as well as the stochastic-volatility version.",
        forecast_shape="A flat line at the current estimated trend — smoother and less jumpy than the random walk.",
    )

    def __init__(self, stochastic_vol: bool = False):
        super().__init__(stochastic_vol=stochastic_vol)

    def _fit(self) -> None:
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        y = self._y.reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = UnobservedComponents(y, level="local level")
            self._res = mod.fit(disp=False)
        self._trend = float(np.asarray(self._res.smoothed_state[0])[-1])

    def _forecast(self, h: int) -> float:
        return self._trend


# --------------------------------------------------------------------------- #
# Full stochastic-volatility UCSV via Gibbs sampling
# --------------------------------------------------------------------------- #
def _ffbs_local_level(y, v_obs, v_state, rng):
    """Forward-filter backward-sample the trend of a local-level model with
    *time-varying* observation variances `v_obs` (sigma_eps_t^2) and state
    innovation variances `v_state` (sigma_eta_t^2). Returns a sampled trend path."""
    n = len(y)
    a = np.empty(n); R = np.empty(n); m = np.empty(n); C = np.empty(n)

    # diffuse-ish initialization at t=0
    m[0] = y[0]
    C[0] = 1e4
    # update at t=0 with the observation
    Q0 = C[0] + v_obs[0]
    m[0] = m[0] + (C[0] / Q0) * (y[0] - m[0])
    C[0] = C[0] * v_obs[0] / Q0
    a[0] = m[0]; R[0] = C[0]

    for t in range(1, n):
        a[t] = m[t - 1]
        R[t] = C[t - 1] + v_state[t]
        Q = R[t] + v_obs[t]
        gain = R[t] / Q
        m[t] = a[t] + gain * (y[t] - a[t])
        C[t] = R[t] * v_obs[t] / Q

    tau = np.empty(n)
    tau[n - 1] = m[n - 1] + np.sqrt(max(C[n - 1], 0.0)) * rng.standard_normal()
    for t in range(n - 2, -1, -1):
        B = C[t] / R[t + 1]
        h_mean = m[t] + B * (tau[t + 1] - a[t + 1])
        h_var = max(C[t] - B * B * R[t + 1], 0.0)
        tau[t] = h_mean + np.sqrt(h_var) * rng.standard_normal()
    return tau


def _sv_single_move(resid, h, gamma2, rng):
    """One Gibbs sweep of the log-volatility path `h` for a series `resid` whose
    conditional variance is exp(h_t), with a random-walk law of motion for h and
    innovation variance `gamma2`. Jacquier–Polson–Rossi single-move Metropolis:
    the proposal is the (Gaussian) state prior, so acceptance uses only the
    measurement likelihood ratio. Updates `h` in place."""
    n = len(h)
    r2 = resid * resid
    for t in range(n):
        if t == 0:
            mu = h[1] if n > 1 else h[0]
            var = gamma2
        elif t == n - 1:
            mu = h[t - 1]
            var = gamma2
        else:
            mu = 0.5 * (h[t - 1] + h[t + 1])
            var = 0.5 * gamma2
        prop = mu + np.sqrt(var) * rng.standard_normal()
        # log-likelihood of resid_t under N(0, exp(h)):  -h/2 - 0.5 r2 exp(-h)
        ll_prop = -0.5 * prop - 0.5 * r2[t] * np.exp(-prop)
        ll_cur = -0.5 * h[t] - 0.5 * r2[t] * np.exp(-h[t])
        if np.log(rng.random()) < (ll_prop - ll_cur):
            h[t] = prop


class UCSVSV(ForecastModel):
    info = ModelInfo(
        key="ucsvsv",
        name="UCSV-SV (stochastic volatility, MCMC)",
        family="Statistical",
        reference="Stock–Watson (2007)",
        description=(
            "The full Stock–Watson unobserved-components stochastic-volatility model, "
            "estimated by Gibbs sampling: the random-walk trend is drawn by a "
            "forward-filter-backward-sample step, and the two log-volatility paths "
            "(for the trend and transitory shocks) by a single-move sampler. The "
            "forecast is the posterior-mean trend. This is the modern benchmark for "
            "'underlying' inflation. Slower than the local-level UCSV because it runs "
            "an MCMC chain per fit."
        ),
        citation="Stock, J. & Watson, M. (2007), 'Why Has US Inflation Become Harder to Forecast?', JMCB — the full UCSV model.",
        intuition="Like UCSV, but it also learns how *noisy* inflation is at each date, so it trusts recent data more in calm periods and less in volatile ones.",
        unique="The only model here with time-varying volatility: it adapts how much a new print moves the trend depending on the era (Great Moderation vs. 2021–22).",
        strengths="The canonical explanation for why inflation became 'harder to forecast' — matches the rise of trend volatility in the 1970s and post-2020.",
        caveats="Requires an MCMC chain per fit (~seconds), so it is heavier than the other statistical models; still a flat-trend point forecast.",
        forecast_shape="A flat line at the posterior-mean trend, with the model also reporting how uncertain that trend is.",
    )

    def __init__(self, n_draws: int = 400, burn: int = 400, gamma: float = 0.2,
                 thin: int = 1, seed: int = 12345):
        super().__init__(n_draws=n_draws, burn=burn, gamma=gamma, seed=seed)
        self.n_draws = n_draws
        self.burn = burn
        self.gamma2 = float(gamma) ** 2   # variance of the log-vol innovations
        self.thin = max(1, thin)
        self.seed = seed

    def _fit(self) -> None:
        y = self._y.values.astype(float)
        n = len(y)
        rng = np.random.default_rng(self.seed)

        # --- initialize volatility paths from a rough decomposition -----------
        # crude trend via centered MA to seed residual scales
        k = min(13, n)
        trend0 = np.convolve(y, np.ones(k) / k, mode="same")
        eps0 = y - trend0
        eta0 = np.diff(trend0, prepend=trend0[0])
        v_eps = np.log(np.maximum(eps0 ** 2, 1e-4))
        v_eta = np.log(np.maximum(eta0 ** 2, 1e-4))
        # smooth the seeds a little
        v_eps = np.convolve(v_eps, np.ones(k) / k, mode="same")
        v_eta = np.convolve(v_eta, np.ones(k) / k, mode="same")

        trend_draws = np.empty(self.n_draws)
        # store full posterior-mean paths for diagnostics/plotting
        tau_sum = np.zeros(n)
        sig_eps_sum = np.zeros(n)
        sig_eta_sum = np.zeros(n)
        kept = 0

        total = self.burn + self.n_draws * self.thin
        for it in range(total):
            # 1) draw trend given volatility paths
            tau = _ffbs_local_level(y, np.exp(v_eps), np.exp(v_eta), rng)
            # 2) residuals and volatility updates
            eps = y - tau
            eta = np.diff(tau, prepend=tau[0])   # eta_0 defined as 0-change
            _sv_single_move(eps, v_eps, self.gamma2, rng)
            _sv_single_move(eta, v_eta, self.gamma2, rng)

            if it >= self.burn and (it - self.burn) % self.thin == 0:
                trend_draws[kept] = tau[-1]
                tau_sum += tau
                sig_eps_sum += np.exp(0.5 * v_eps)
                sig_eta_sum += np.exp(0.5 * v_eta)
                kept += 1

        self._trend = float(np.mean(trend_draws[:kept]))
        self._trend_sd = float(np.std(trend_draws[:kept]))
        self.trend_path_ = tau_sum / kept
        self.sigma_eps_path_ = sig_eps_sum / kept
        self.sigma_eta_path_ = sig_eta_sum / kept

    def _forecast(self, h: int) -> float:
        # random-walk trend ⇒ flat forecast at the posterior-mean trend for all h
        return self._trend
