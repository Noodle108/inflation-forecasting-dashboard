"""Stock–Watson multi-sector Dynamic Factor Model.

Stock & Watson's multi-sector approach to inflation — refined across a series of
papers culminating in Stock & Watson (2016, "Core Inflation and Trend Inflation") —
disaggregates headline CPI into its sectoral subcomponents (apparel, food, energy,
housing, transport, medical, recreation, education & communication, other services)
and extracts a **common factor** driving them all. The forecastable part of headline
inflation is then the piece explained by that common factor plus the persistence of
the aggregate, with sector-idiosyncratic movements averaged out.

Implementation follows Stock–Watson's diffusion-index *direct forecast*:

1. build sectoral inflation from the FRED CPI subindices (annualized log-differences);
2. standardize and extract the first principal component F_t as the common factor
   (equivalent, up to sign, to the single-factor DFM estimate under the exact static
   factor model — Stock & Watson (2002) show PC estimates the factor consistently);
3. estimate one h-step regression per horizon,
       pi_{t+h} = μ_h + α_h · pi_t + β_h · F_t + eps,
   fitted separately for each h. Direct forecasting sidesteps the near-collinearity
   between F_t and π_t (they are both aggregates of the same subindices) that makes
   an iterated factor-augmented VAR unstable, and is the specification used in the
   forecasting-side Stock–Watson papers.

The result is a sector-diversified forecast: robust to a shock in any one component
(e.g. a gasoline spike, a used-car surge) because the common factor is the pooled
signal across all nine CPI groups.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import ForecastModel, ModelInfo

# CPI major-group subindices on FRED. These 8 groups are the ones with long, clean
# monthly histories (CPIRECSL begins 1993; the others go back to the 1950s).
_CPI_SECTORS = dict(
    apparel="CPIAPPSL",
    food="CPIUFDSL",
    energy="CPIENGSL",
    housing="CPIHOSSL",
    transport="CPITRNSL",
    medical="CPIMEDSL",
    recreation="CPIRECSL",
    education="CPIEDUSL",
)


def _load_sector_panel(freq: str = "M", start: str = "1990-01-01") -> pd.DataFrame:
    """Panel of annualized inflation for CPI subcomponents."""
    from ..data import fred as _f
    if not _f.has_live_data():
        raise RuntimeError("Stock–Watson DFM needs live FRED data.")
    factor = 1200.0 if freq == "M" else 400.0
    cols = {}
    for name, sid in _CPI_SECTORS.items():
        s = _f._fetch_fred_series(sid, start)
        if s is None:
            continue
        if freq == "Q":
            s = s.resample("QS").mean()
        infl = factor * np.log(s / s.shift(1))
        cols[name] = infl
    if not cols:
        raise RuntimeError("Stock–Watson DFM: no CPI subindices returned.")
    return pd.DataFrame(cols).dropna()


class StockWatsonDFM(ForecastModel):
    info = ModelInfo(
        key="swdfm",
        name="Stock–Watson multi-sector DFM",
        family="Statistical",
        reference="Stock–Watson (2002, 2016) multi-sector DFM",
        description=(
            "A dynamic factor model over the eight major CPI subcomponents (apparel, "
            "food, energy, housing, transport, medical, recreation, education). The "
            "first principal component of standardized sectoral inflation is extracted "
            "as the common factor F_t; headline inflation is then forecast by a "
            "factor-augmented AR — lagged inflation plus lagged F_t — with the factor "
            "iterated forward as an AR(1). Diversifies across sectors, so a shock in "
            "any single component (energy, used cars) is averaged out."
        ),
        needs_activity=False,
        citation="Stock, J. & Watson, M. (2002), 'Macroeconomic Forecasting Using Diffusion Indexes', JBES 20(2); Stock, J. & Watson, M. (2016), 'Core Inflation and Trend Inflation', REStat 98(4).",
        intuition="Averages the eight major CPI sectors into a single 'diffusion index' that captures common inflationary pressure, then rides that factor forward — sectoral idiosyncrasies (gasoline spikes, used-car surges) wash out.",
        unique="The only model here that reads disaggregated CPI data: it distinguishes a broad-based inflationary impulse (factor rising across all sectors) from a narrow one (only energy) — precisely the signal the Fed watches in the sectoral CPI decomposition.",
        strengths="Robust to any single-sector shock; extracts common inflation signal exactly the way FOMC briefing books decompose the CPI.",
        caveats="Static single-factor approximation to the true dynamic factor model (Stock–Watson use one factor in most applications, so this is close to the paper's setup); PC-based estimation ignores measurement-error heteroskedasticity.",
        forecast_shape="Mean-reverting: the factor decays back to its mean while inflation drifts toward the sample average — sloped path with speed set by factor persistence.",
    )

    def __init__(self, max_h: int = 24):
        super().__init__(max_h=max_h)
        self.max_h = max_h

    def _fit(self) -> None:
        import statsmodels.api as sm

        is_quarterly = bool(self._y.index.freqstr and self._y.index.freqstr.startswith("Q"))
        freq = "Q" if is_quarterly else "M"
        panel = _load_sector_panel(freq=freq)
        panel = panel.loc[:self._y.index[-1]]
        y = self._y.loc[:panel.index[-1]].reindex(panel.index).dropna()
        panel = panel.reindex(y.index)

        # Standardize; extract F_t as the first principal component (PCA via SVD).
        Z = panel.values
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Zs = (Z - mu) / sd
        U, S, Vt = np.linalg.svd(Zs, full_matrices=False)
        F = U[:, 0] * S[0] / np.sqrt(len(Zs))
        sign = 1.0 if np.corrcoef(F, y.values)[0, 1] >= 0 else -1.0
        F = sign * F
        F_ser = pd.Series(F, index=y.index, name="F")
        self._loadings = pd.Series(Vt[0] * sign, index=panel.columns)
        self._factor = F_ser

        # ---- Direct h-step forecasting regressions, one per horizon 1..H ----
        # pi_{t+h} = μ_h + α_h · pi_t + β_h · F_t   (Stock–Watson diffusion index)
        self._direct = {}
        d0 = pd.DataFrame({"pi": y, "F": F_ser})
        for h in range(1, self.max_h + 1):
            d = d0.copy()
            d["y"] = d0["pi"].shift(-h)
            dd = d.dropna()
            if len(dd) < 30:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Xp = sm.add_constant(dd[["pi", "F"]].values)
                self._direct[h] = sm.OLS(dd["y"].values, Xp).fit()

        self._last_pi = float(y.iloc[-1])
        self._last_F = float(F_ser.iloc[-1])

    def _forecast(self, h: int) -> float:
        # Use the direct regression for horizon h if available, otherwise the closest.
        h_use = h if h in self._direct else max(k for k in self._direct if k <= h)
        c, a, b = self._direct[h_use].params
        return float(c + a * self._last_pi + b * self._last_F)
