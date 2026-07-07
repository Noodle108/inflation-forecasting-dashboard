"""Per-model data-source / assumptions / equations metadata.

Kept in a single file (rather than sprinkled across every model module) so all three
fields for all 18 models can be reviewed and edited together.

At import time, :func:`apply_extras` merges these dicts into each ``ModelInfo`` on the
registered model classes — read-only augmentation, no behavior change.
"""
from __future__ import annotations

# FRED series page templates — keeps individual entries short and links stable.
def _fred(sid: str, label: str | None = None) -> tuple[str, str]:
    return (label or f"FRED: {sid}", f"https://fred.stlouisfed.org/series/{sid}")


# --------------------------------------------------------------------------- #
# Per-model metadata
# --------------------------------------------------------------------------- #
EXTRAS: dict[str, dict] = {
    # ----- Benchmarks -----
    "rw": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("CPILFESL", "Core CPI"),
            _fred("PCEPI", "PCE (headline)"),
            _fred("PCEPILFE", "Core PCE"),
        ],
        assumptions=(
            "Inflation follows an integrated (unit-root) process, so today's rate is "
            "the best guess of tomorrow's — no mean-reversion. Equivalent to assuming "
            "trend inflation is a random walk (Atkeson–Ohanian, IMA(1,1))."
        ),
        equations=r"\pi_{t+h|t} \;=\; \pi_t",
    ),
    "ao": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("CPILFESL", "Core CPI"),
            _fred("PCEPI", "PCE (headline)"),
            _fred("PCEPILFE", "Core PCE"),
        ],
        assumptions=(
            "Best point forecast of next year's inflation is the trailing four-quarter "
            "(twelve-month) average — a smoother random walk. No structural relation to "
            "the real economy."
        ),
        equations=r"\pi_{t+h|t} \;=\; \frac{1}{W}\sum_{i=0}^{W-1}\pi_{t-i}\quad(W{=}12\text{ or }4)",
    ),

    # ----- Statistical -----
    "ar": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("CPILFESL", "Core CPI"),
            _fred("PCEPI", "PCE (headline)"),
            _fred("PCEPILFE", "Core PCE"),
        ],
        assumptions=(
            "Inflation is a stationary linear autoregression of finite order p; "
            "coefficients are constant across the sample; shocks are homoskedastic and "
            "uncorrelated. Lag order p is picked by BIC (or specified explicitly)."
        ),
        equations=r"\pi_t \;=\; c \;+\; \sum_{i=1}^{p}\phi_i \pi_{t-i} \;+\; \varepsilon_t,"
                   r"\qquad p\ \text{chosen by BIC}\ \le\ p_{\max}",
    ),
    "pc": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("UNRATE", "Unemployment rate"),
        ],
        assumptions=(
            "Accelerationist (backward-looking) Phillips curve: inflation responds to "
            "its own recent history and to labor-market slack. Slack measured by the "
            "unemployment gap (u_t − trend), an NAIRU proxy. Direct h-step regression, "
            "so expectations are handled reduced-form."
        ),
        equations=r"\pi_{t+h}\;=\;c\;+\;\sum_{i=0}^{p-1}\alpha_i\pi_{t-i}\;+\;\sum_{i=0}^{p-1}\beta_i\,\tilde u_{t-i}\;+\;\varepsilon_{t+h}",
    ),
    "ucsv": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("CPILFESL", "Core CPI"),
            _fred("PCEPI", "PCE (headline)"),
            _fred("PCEPILFE", "Core PCE"),
        ],
        assumptions=(
            "Inflation = slow-moving stochastic trend τ_t + transitory noise ε_t. Trend "
            "is a Gaussian random walk; noise is iid Gaussian. Variances are constant "
            "across the sample (the 'no-SV' approximation to Stock–Watson 2007)."
        ),
        equations=(r"\pi_t \;=\; \tau_t \;+\; \varepsilon_t,\quad "
                   r"\tau_t \;=\; \tau_{t-1}\;+\;\eta_t;\qquad "
                   r"\pi_{t+h|t}\;=\;\tau_T"),
    ),
    "ucsvsv": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("CPILFESL", "Core CPI"),
            _fred("PCEPI", "PCE (headline)"),
            _fred("PCEPILFE", "Core PCE"),
        ],
        assumptions=(
            "Same trend + transitory decomposition as UCSV, but both innovation "
            "variances follow their own random walks in log space (stochastic "
            "volatility). Estimated by Gibbs sampling; the point forecast is the "
            "posterior-mean trend."
        ),
        equations=(r"\pi_t=\tau_t+\varepsilon_t,\ \varepsilon_t\!\sim\!N(0,\sigma_{\varepsilon,t}^2);\quad "
                   r"\tau_t=\tau_{t-1}+\eta_t,\ \eta_t\!\sim\!N(0,\sigma_{\eta,t}^2);\quad "
                   r"\log\sigma_{\cdot,t}^2=\log\sigma_{\cdot,t-1}^2+\nu_{\cdot,t}"),
    ),
    "bvar": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("UNRATE", "Unemployment rate"),
        ],
        assumptions=(
            "A finite-order linear VAR(p) in [inflation, unemployment] with constant "
            "coefficients. Bayesian Minnesota prior shrinks each equation toward an "
            "independent random walk (own first lag = 1, others = 0, tighter at longer "
            "lags). Shrinkage λ is fixed exogenously."
        ),
        equations=(r"Y_t=[\pi_t,u_t]';\quad Y_t=c+\sum_{l=1}^{p}A_l Y_{t-l}+\varepsilon_t;\ "
                   r"\text{Minnesota prior: }A_l^{(ii)}\sim N(\mathbb{1}_{l{=}1},(\lambda/l)^2)"),
    ),
    "bvarh": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("UNRATE", "Unemployment rate"),
        ],
        assumptions=(
            "Same BVAR as above, but the Minnesota shrinkage λ is a hyperparameter and "
            "is chosen to maximize the marginal likelihood, with a Gamma hyperprior "
            "(Giannone–Lenza–Primiceri 2015). Data-driven shrinkage."
        ),
        equations=(r"\hat\lambda=\arg\max_\lambda\;p(Y\mid\lambda)\,p(\lambda),\quad "
                   r"p(\lambda)\sim\mathrm{Gamma}(a,b);\ \text{rest as BVAR}"),
    ),
    "tvpvar": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("UNRATE", "Unemployment rate"),
        ],
        assumptions=(
            "VAR(p) in [inflation, unemployment] whose coefficients drift as a Gaussian "
            "random walk over time, and whose log-variances also follow random walks "
            "(stochastic volatility). Estimated by Gibbs (Carter–Kohn for coefficients, "
            "single-move sampler for volatilities). Forecast uses the end-of-sample "
            "coefficient draw."
        ),
        equations=(r"Y_t=X_t'\beta_t+\varepsilon_t,\ \varepsilon_t\!\sim\!N(0,e^{h_t});\ "
                   r"\beta_t=\beta_{t-1}+u_t;\ h_t=h_{t-1}+\nu_t"),
    ),
    "swdfm": dict(
        data_sources=[
            _fred("CPIAPPSL", "CPI: Apparel"),
            _fred("CPIUFDSL", "CPI: Food"),
            _fred("CPIENGSL", "CPI: Energy"),
            _fred("CPIHOSSL", "CPI: Housing"),
            _fred("CPITRNSL", "CPI: Transport"),
            _fred("CPIMEDSL", "CPI: Medical"),
            _fred("CPIRECSL", "CPI: Recreation"),
            _fred("CPIEDUSL", "CPI: Education & Communication"),
            _fred("CPIAUCSL", "Headline CPI (target)"),
        ],
        assumptions=(
            "A single common factor drives comovement across the eight major CPI "
            "sectors (static approximate factor model). PCA is a consistent estimator "
            "of the factor when the number of sectors and observations are both large "
            "(Stock–Watson 2002). Forecast uses a *direct* h-step regression per horizon."
        ),
        equations=(r"X_{it}=\lambda_i F_t+e_{it}\ (i=1..N);\quad "
                   r"\pi_{t+h}=\mu_h+\alpha_h\pi_t+\beta_h F_t+\varepsilon_{t+h}"),
    ),
    "clenow": dict(
        data_sources=[
            _fred("CPIAUCSL", "Headline CPI (target)"),
            _fred("GASREGW", "Retail gasoline, weekly"),
            _fred("CPIUFDSL", "CPI: Food"),
        ],
        assumptions=(
            "Bridge assumption: monthly gasoline price growth (aggregated from the "
            "weekly retail series) is a leading indicator of the energy component of "
            "the monthly CPI print, and food-price growth captures another slow-moving "
            "chunk. Beyond h=1, gasoline and food growth are projected forward as "
            "univariate AR(1)s."
        ),
        equations=(r"\pi_t \;=\; c \;+\; \delta\,g^{\text{gas}}_t \;+\; \phi\,g^{\text{food}}_t"
                   r"\;+\; \sum_{i=1}^{p}\alpha_i\pi_{t-i}\;+\;\sum_{i=1}^{p}\beta_i g^{\text{gas}}_{t-i}\;+\;\varepsilon_t"),
    ),

    # ----- Structural -----
    "nkpc": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("UNRATE", "Unemployment rate"),
        ],
        assumptions=(
            "Hybrid New Keynesian Phillips curve: prices are set forward-looking under "
            "Calvo pricing, so inflation depends on expected next-period inflation and "
            "the output gap. Expectations are approximated by a lagged inflation term "
            "(reduced-form hybrid form, Galí–Gertler 1999). Output gap = Okun-mapped "
            "unemployment gap. No explicit real-wage or Fisher equation."
        ),
        equations=(r"\pi_t \;=\; c\;+\;\beta\,E_t\pi_{t+1}\;+\;\kappa x_t\;+\;u_t\ "
                   r"\rightarrow\ \pi_t\;=\;c\;+\;b\pi_{t-1}\;+\;\kappa x_t\;+\;\varepsilon_t"),
    ),
    "tvtnkpc": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("UNRATE", "Unemployment rate"),
        ],
        assumptions=(
            "Inflation = drifting stochastic trend τ_t (random walk) + stationary "
            "'gap' that obeys a Phillips-curve equation in the output gap. Trend and "
            "gap estimated sequentially (Kalman local level for τ, OLS for the gap). "
            "No Fisher or real-wage equation; inflation is not restricted to be a "
            "function of monetary policy."
        ),
        equations=(r"\pi_t=\tau_t+\tilde\pi_t,\quad \tau_t=\tau_{t-1}+\eta_t,\quad "
                   r"\tilde\pi_t=\rho\tilde\pi_{t-1}-\lambda\,\tilde u_t+\varepsilon_t"),
    ),
    "dsge": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("UNRATE", "Unemployment rate"),
        ],
        assumptions=(
            "Three-equation New Keynesian DSGE with rational expectations. "
            "**IS curve**: consumption Euler equation with log utility (σ=1) — no habits, "
            "no capital. **NKPC**: Calvo pricing with κ=0.05, no indexation. **Taylor rule**: "
            "φ_π=1.5, φ_x=0.125, no interest smoothing. Fisher equation is imposed via "
            "the ex-ante real rate in the IS curve. Real wages implicit (labor supply "
            "not modeled). Shocks are AR(1); β=0.99, so the model imposes an "
            "intertemporal budget constraint but no fiscal block."
        ),
        equations=(r"x_t=E_tx_{t+1}-\tfrac{1}{\sigma}(i_t-E_t\pi_{t+1})+a_t\quad(\text{IS/Fisher})\\ "
                   r"\pi_t=\beta E_t\pi_{t+1}+\kappa x_t+u_t\quad(\text{NKPC})\\ "
                   r"i_t=\phi_\pi\pi_t+\phi_x x_t+v_t\quad(\text{Taylor})"),
    ),
    "sw07": dict(
        data_sources=[
            _fred("GDPC1", "Real GDP"),
            _fred("PCECC96", "Real consumption"),
            _fred("GPDIC1", "Real investment"),
            _fred("GDPDEF", "GDP deflator (inflation target)"),
            _fred("HOANBS", "Hours worked"),
            _fred("COMPRNFB", "Real compensation"),
            _fred("FEDFUNDS", "Federal funds rate"),
            _fred("CNP16OV", "Civilian population 16+"),
        ],
        assumptions=(
            "Full Smets–Wouters (2007) medium-scale NK DSGE. "
            "**Households**: external habit in consumption (h=0.71), separable labor "
            "supply with Frisch elasticity 1/σ_l=1/1.83. **Firms**: Calvo prices "
            "(θ_p=0.66) and Calvo wages (θ_w=0.70) with partial indexation. "
            "**Real wages** move with productivity and price shocks via a New Keynesian "
            "wage curve; wages are not competitively priced. **Fisher equation** "
            "imposed on ex-ante real rates in the Euler equation and Tobin's-q. "
            "**Capital** has adjustment costs and variable utilization. **Monetary "
            "policy**: inertial Taylor rule with weight on inflation, output level, "
            "and output growth. Rational expectations; log-linear approximation around "
            "the balanced growth path. Deep parameters fixed at the 2007 posterior mode."
        ),
        equations=r"\text{33 log-linearized equations; core: NKPC, wage curve, Euler with real rate, Taylor rule, capital accumulation.}",
    ),
    "bb": dict(
        data_sources=[
            _fred("CPIAUCSL", "Headline CPI"),
            _fred("CPIENGSL", "CPI: Energy"),
            _fred("CPIUFDSL", "CPI: Food"),
            _fred("ECIWAG", "ECI: wages & salaries"),
            _fred("JTSJOL", "JOLTS: job openings"),
            _fred("UNEMPLOY", "Unemployed persons"),
            _fred("MICH", "Michigan 1-year expected inflation"),
            _fred("EXPINF10YR", "Cleveland Fed 10-year expected inflation"),
        ],
        assumptions=(
            "Semi-structural wage–price system. **Price equation**: mark-up pricing with "
            "unit long-run pass-through of wages to prices — no separate profit share, "
            "no explicit real wage. **Wage equation**: wage Phillips curve driven by "
            "labor-market tightness (v/u), a catch-up term for past realized vs. "
            "expected inflation, and short-run expectations. Long-run **anchoring**: both "
            "equations estimated in expectation-gap form so wages and prices converge "
            "to expected inflation in steady state. Expectations are taken from surveys "
            "(reduced-form, not rational). Omits the shortages term (not on FRED)."
        ),
        equations=(r"g^p_t-E^{LR}_t=c+\sum a_i(g^p_{t-i}-E^{LR}_t)+b(g^w_t-E^{LR}_t)+g_e r^e_t+g_f r^f_t\\ "
                   r"g^w_t-E^{SR}_t=c+\sum e_i(g^w_{t-i}-E^{SR}_t)+f\,v/u_t+h(g^p_{t-1}-E^{SR}_{t-1})"),
    ),
    "nyfed": dict(
        data_sources=[
            _fred("CPIAUCSL", "CPI (headline)"),
            _fred("UNRATE", "Unemployment rate"),
            _fred("BAA10YM", "BAA–10Y credit spread"),
        ],
        assumptions=(
            "Small-scale NK DSGE augmented with a Bernanke–Gertler–Gilchrist financial "
            "wedge. Rational expectations. Fisher equation imposed in the IS curve. "
            "**Financial friction**: a credit-spread shock b_t shifts spending in the "
            "IS curve with the opposite sign of a demand shock; it is identified from "
            "the BAA–10Y spread through a linear observation equation. Real wages "
            "implicit. Deep params calibrated (β=0.99, σ=1, κ=0.05, φ_π=1.5); the four "
            "shock persistences, variances, and the spread-loading θ are estimated by "
            "Kalman-filter maximum likelihood."
        ),
        equations=(r"x_t=E_tx_{t+1}-\tfrac{1}{\sigma}(i_t-E_t\pi_{t+1})+a_t-b_t\\ "
                   r"\pi_t=\beta E_t\pi_{t+1}+\kappa x_t+u_t\\ "
                   r"i_t=\phi_\pi\pi_t+\phi_x x_t+v_t\\ "
                   r"\text{spread}_t=\theta b_t+\text{m.err.}"),
    ),
    "cleexp": dict(
        data_sources=[
            ("FRED: EXPINF1YR .. EXPINF30YR (Cleveland Fed term structure)",
             "https://fred.stlouisfed.org/searchresults/?st=EXPINF&pageID=1"),
            ("Cleveland Fed: inflation expectations methodology",
             "https://www.clevelandfed.org/indicators-and-data/inflation-expectations"),
        ],
        assumptions=(
            "The Haubrich–Pennacchi–Ritchken (2012) affine term-structure model jointly "
            "fits nominal Treasuries, TIPS, inflation swaps, and survey expectations. "
            "Key identifying assumptions: (i) **absence of arbitrage** across nominal "
            "and real markets; (ii) an **affine SDF** so real rates, expected "
            "inflation, and their risk premia are all affine functions of latent "
            "factors; (iii) the **Fisher equation** links nominal and real rates "
            "period-by-period. The Cleveland Fed does the estimation; this model is a "
            "pass-through wrapper that returns the horizon-h expected-inflation average."
        ),
        equations=(r"i^{\$}_{t,n}=i^{r}_{t,n}+E_t\bar\pi_{t,n}+\text{IRP}_{t,n};\ "
                   r"\pi_{t+h|t}=E_t\bar\pi_{t,h}\text{ from the fitted curve}"),
    ),

    "cpi_data": None,  # placeholder — never registered, just to keep this dict tidy
}


def apply_extras(models_dict: dict) -> None:
    """Merge EXTRAS onto each ``ModelInfo`` in a {key -> factory} registry."""
    for key, factory in models_dict.items():
        extra = EXTRAS.get(key)
        if not extra:
            continue
        info = factory().info   # ModelInfo is a class attribute shared per model class
        if "data_sources" in extra:
            info.data_sources = list(extra["data_sources"])
        if "assumptions" in extra:
            info.assumptions = extra["assumptions"]
        if "equations" in extra:
            info.equations = extra["equations"]
