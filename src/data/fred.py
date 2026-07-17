"""Data access layer.

Pulls macro series from FRED, converts price indices into annualized inflation,
and assembles the tidy dataset the models consume. If no FRED API key is available
(or the network is down) it falls back to a deterministic *synthetic* dataset so the
whole pipeline — models, backtest, dashboard — runs offline.

Inflation convention
--------------------
For a monthly price index ``P_t`` we use the annualized log change

    pi_t = 1200 * ln(P_t / P_{t-1})          (monthly series)
    pi_t =  400 * ln(P_t / P_{t-1})          (quarterly series)

which is the standard convention in Stock–Watson and most of the forecasting
literature.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Project root = two levels up from this file (src/data/fred.py -> dashboard/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load the .env from the project root by *absolute path* so it is found no matter
# what the current working directory is when the app is launched. (Streamlit's CWD
# is wherever `streamlit run` was invoked, which is often not the project root — the
# bare load_dotenv() default searches the CWD and silently misses the key, causing a
# fallback to synthetic data.)
try:  # optional dependency / optional key
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Series catalog: the price measures and activity variables used by the models.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PriceSeries:
    key: str          # short internal id
    fred_id: str      # FRED series id (price index, level)
    label: str        # human label
    freq: str         # 'M' or 'Q'


PRICE_SERIES = {
    "cpi": PriceSeries("cpi", "CPIAUCSL", "CPI (headline)", "M"),
    "core_cpi": PriceSeries("core_cpi", "CPILFESL", "Core CPI (ex food & energy)", "M"),
    "pce": PriceSeries("pce", "PCEPI", "PCE (headline)", "M"),
    "core_pce": PriceSeries("core_pce", "PCEPILFE", "Core PCE (ex food & energy)", "M"),
}

# Activity / slack variables used by the Phillips curve and (later) BVAR/DSGE.
ACTIVITY_SERIES = {
    "unrate": "UNRATE",        # unemployment rate
    "ngap": None,              # unemployment gap, derived below
}

DEFAULT_START = "1960-01-01"

# Annualization factor by frequency.
_ANNUALIZE = {"M": 1200.0, "Q": 400.0}


# ---------------------------------------------------------------------------
# FRED access
# ---------------------------------------------------------------------------
def _fred_client():
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        return None
    try:
        from fredapi import Fred

        return Fred(api_key=key)
    except Exception:
        return None


def has_live_data() -> bool:
    """True if a FRED key is configured and the client initializes."""
    return _fred_client() is not None


@lru_cache(maxsize=32)
def _fetch_fred_series(fred_id: str, start: str) -> Optional[pd.Series]:
    client = _fred_client()
    if client is None:
        return None
    try:
        s = client.get_series(fred_id, observation_start=start)
        s.index = pd.to_datetime(s.index)
        return s.astype(float).dropna()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ALFRED (real-time / vintage) access
# ---------------------------------------------------------------------------
_ALFRED_CACHE_DIR = PROJECT_ROOT / "data" / "alfred"
_ALFRED_MEM: dict[str, pd.DataFrame] = {}


def _fetch_alfred_all_releases(fred_id: str) -> Optional[pd.DataFrame]:
    """Return every ALFRED release row for ``fred_id``: (realtime_start, date, value).

    Cached on disk under ``data/alfred/<fred_id>.parquet`` so repeated runs don't
    re-hit FRED. Values are kept in their raw units (levels), same as the FRED
    published series. Rows with ``value = NaT`` (ALFRED's marker for "no
    observation reported in that vintage") are dropped.
    """
    if fred_id in _ALFRED_MEM:
        return _ALFRED_MEM[fred_id]
    _ALFRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _ALFRED_CACHE_DIR / f"{fred_id}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        client = _fred_client()
        if client is None:
            return None
        try:
            df = client.get_series_all_releases(fred_id)
        except Exception:
            return None
        df["realtime_start"] = pd.to_datetime(df["realtime_start"])
        df["date"] = pd.to_datetime(df["date"])
        # Filter out NaT-value rows; keep dtype float
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df.to_parquet(cache)
    _ALFRED_MEM[fred_id] = df
    return df


def fetch_vintage_series(fred_id: str, as_of: pd.Timestamp,
                         fallback_start: str = "1960-01-01") -> Optional[pd.Series]:
    """Return ``fred_id`` as it was known on ``as_of``.

    For each observation date, uses the *latest* revision released on or before
    ``as_of``. If the series has no ALFRED history before ``as_of`` (e.g. GDPDEF
    before 1991-12), falls back to the current-vintage series (with observations
    beyond ``as_of`` clipped so we still can't peek at future data).
    """
    df = _fetch_alfred_all_releases(fred_id)
    if df is None or df.empty:
        return None
    v = df[df["realtime_start"] <= pd.Timestamp(as_of)]
    if v.empty:
        latest = _fetch_fred_series(fred_id, fallback_start)
        if latest is None:
            return None
        return latest[latest.index <= pd.Timestamp(as_of)]
    # Latest revision per observation date, as of `as_of`.
    v = v.sort_values("realtime_start").drop_duplicates("date", keep="last")
    s = pd.Series(v["value"].values, index=v["date"]).sort_index()
    # An observation only appears in ALFRED after it was first released; nothing
    # beyond as_of is knowable. This filter is redundant given the realtime_start
    # cut but makes the invariant explicit.
    return s[s.index <= pd.Timestamp(as_of)]


def to_inflation(level: pd.Series, freq: str) -> pd.Series:
    """Annualized log-difference inflation from a price *level* series."""
    factor = _ANNUALIZE[freq]
    pi = factor * np.log(level / level.shift(1))
    return pi.dropna()


# ---------------------------------------------------------------------------
# Synthetic fallback (offline mode)
# ---------------------------------------------------------------------------
def _synthetic_dataset(freq: str = "M", n: int = 720, seed: int = 7) -> "MacroData":
    """A UCSV-flavored synthetic world: a slow random-walk trend + stochastic-vol
    noise, plus an unemployment series negatively correlated with inflation gaps.
    Deterministic given the seed so results are reproducible."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1965-01-01", periods=n, freq="MS" if freq == "M" else "QS")

    # trend inflation as a random walk with time-varying innovation vol
    log_sig_eta = np.cumsum(rng.normal(0, 0.15, n)) * 0.05 - 1.6
    tau = np.cumsum(rng.normal(0, np.exp(log_sig_eta))) + 2.5
    log_sig_eps = np.cumsum(rng.normal(0, 0.2, n)) * 0.05 - 0.2
    eps = rng.normal(0, np.exp(log_sig_eps))
    pi = tau + eps

    # unemployment: mean-reverting, pushed up when inflation runs below trend
    u = np.zeros(n)
    u[0] = 5.5
    for t in range(1, n):
        u[t] = 0.92 * u[t - 1] + 0.08 * 5.5 - 0.10 * (pi[t - 1] - tau[t - 1]) + rng.normal(0, 0.2)

    inflation = pd.DataFrame({"synthetic": pd.Series(pi, index=idx)})
    activity = pd.DataFrame({"unrate": pd.Series(u, index=idx)})
    return MacroData(inflation=inflation, activity=activity, freq=freq, is_synthetic=True)


# ---------------------------------------------------------------------------
# Public dataset object
# ---------------------------------------------------------------------------
@dataclass
class MacroData:
    inflation: pd.DataFrame   # columns = price-measure keys, values = annualized inflation
    activity: pd.DataFrame    # columns = activity keys (unrate, ngap, ...)
    freq: str
    is_synthetic: bool = False

    def series(self, key: str) -> pd.Series:
        """Return a single inflation series, aligned and NaN-dropped."""
        if key not in self.inflation.columns:
            raise KeyError(f"Unknown inflation series '{key}'. "
                           f"Available: {list(self.inflation.columns)}")
        return self.inflation[key].dropna()

    def frame(self, infl_key: str) -> pd.DataFrame:
        """inflation + activity aligned on a common index (inner join, ffill activity)."""
        pi = self.inflation[[infl_key]].rename(columns={infl_key: "pi"})
        act = self.activity.reindex(pi.index).ffill()
        return pi.join(act, how="left").dropna()

    @property
    def price_labels(self) -> dict:
        if self.is_synthetic:
            return {"synthetic": "Synthetic inflation (offline demo)"}
        return {k: PRICE_SERIES[k].label for k in self.inflation.columns}


def load_data(freq: str = "M", start: str = DEFAULT_START) -> MacroData:
    """Load the full macro dataset from FRED, or synthetic data if unavailable."""
    client = _fred_client()
    if client is None:
        return _synthetic_dataset(freq=freq)

    infl = {}
    for key, ps in PRICE_SERIES.items():
        level = _fetch_fred_series(ps.fred_id, start)  # monthly price level
        if level is None or len(level) <= 24:
            continue
        if freq == "Q":
            # aggregate the monthly index to quarterly, then take quarterly inflation
            level = level.resample("QS").mean()
        infl[key] = to_inflation(level, freq)

    if not infl:
        return _synthetic_dataset(freq=freq)

    inflation = pd.DataFrame(infl)

    # activity block
    unrate = _fetch_fred_series("UNRATE", start)
    activity = pd.DataFrame()
    if unrate is not None:
        if freq == "Q":
            unrate = unrate.resample("QS").mean()
        activity["unrate"] = unrate
        # unemployment gap vs a slow-moving (5yr) trend as a simple NAIRU proxy
        window = 60 if freq == "M" else 20
        activity["ngap"] = unrate - unrate.rolling(window, min_periods=12).mean()

    # Long-run expected inflation from the Cleveland Fed term-structure model.
    # This is the anchor Phillips-curve gap models converge to.
    exp10 = _fetch_fred_series("EXPINF10YR", start)
    if exp10 is not None:
        if freq == "Q":
            exp10 = exp10.resample("QS").mean()
        activity["exp10yr"] = exp10

    # Short-run inflation expectations (1y) for models that want a nearer anchor.
    exp1 = _fetch_fred_series("EXPINF1YR", start)
    if exp1 is not None:
        if freq == "Q":
            exp1 = exp1.resample("QS").mean()
        activity["exp1yr"] = exp1

    return MacroData(inflation=inflation, activity=activity.dropna(how="all"), freq=freq)
