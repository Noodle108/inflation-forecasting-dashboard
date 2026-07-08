"""Pass-through forecast models for the three Faust–Wright survey benchmarks.

Faust–Wright's most important empirical finding is that **judgmental survey
forecasts dominate all model-based forecasts** — often by a wide margin. These
three wrappers plug the surveys into the same ForecastModel interface as
everything else, so they show up in the horse race on equal footing.

Which inflation measure to serve
--------------------------------
Both SPF and Greenbook publish forecasts for **multiple inflation measures**
(headline CPI, core CPI, headline PCE, core PCE, GDP deflator). At construction
time the wrapper doesn't know which measure the user has selected; the
Streamlit tab tells it by setting ``model.info.series_key = <key>`` before
``fit`` — or we auto-detect by matching the training-series' *mean* against the
SPF sheets' F0 (nowcast) mean.

Horizon convention
------------------
* SPF: h0..h5 (nowcast through 5-quarters-ahead).
* Greenbook: h0..h9 (nowcast through 9-quarters-ahead).
* Blue-Chip surrogate: single value = 1-year expected inflation, used for every
  horizon (MICH + EXPINF1YR average).

If the horse race asks for a horizon beyond what the survey provides, we return
the largest available horizon.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.surveys import (
    load_blue_chip_surrogate,
    load_greenbook,
    load_spf,
)
from .base import ForecastModel, ModelInfo


# Recognized inflation-measure keys we can serve. Order matters for auto-detect:
# first hit wins if multiple match. Keys must be lowercase.
_INFL_KEYS = ("cpi", "core_cpi", "pce", "core_pce", "gdpdef")


def _guess_series_key(y: pd.Series, X: pd.DataFrame | None) -> str:
    """Best-effort read of which inflation measure ``y`` represents.

    * If the Series has a ``.name`` attribute in ``_INFL_KEYS``, use it.
    * Otherwise fall back to ``"cpi"``.
    """
    name = (y.name or "").lower() if hasattr(y, "name") else ""
    if name in _INFL_KEYS:
        return name
    return "cpi"


class _SurveyBase(ForecastModel):
    """Common lookup logic: pick the most recent survey origin ≤ the last fit-time
    date, then read the horizon-h column."""

    survey_name: str = "Survey"

    def _load(self, series_key: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError

    def _fit(self) -> None:
        series_key = _guess_series_key(self._y, self._X)
        self._series_key = series_key
        w = self._load(series_key)
        if w is None or w.empty:
            self._wide = None
            self._latest = None
            self._last_origin = None
            return
        pi_last = self._y.index[-1]
        w = w.loc[:pi_last]
        self._wide = w
        if w.empty:
            self._latest = None
            self._last_origin = None
        else:
            self._latest = w.iloc[-1]
            self._last_origin = w.index[-1]

    def _forecast(self, h: int) -> float:
        if self._wide is None or self._latest is None:
            return float(self._y.iloc[-1])
        cols = [c for c in self._latest.index if c.startswith("h")]
        # Try the exact horizon; if unavailable fall back to the largest available <=h,
        # then the largest available overall.
        col = f"h{h}"
        val = self._latest.get(col, np.nan)
        if pd.isna(val):
            available = sorted(int(c[1:]) for c in cols)
            le = [x for x in available if x <= h]
            h_use = (max(le) if le else max(available)) if available else None
            if h_use is None:
                return float(self._y.iloc[-1])
            val = self._latest.get(f"h{h_use}", np.nan)
        if pd.isna(val):
            return float(self._y.iloc[-1])
        return float(val)


class SurveySPF(_SurveyBase):
    survey_name = "SPF"
    info = ModelInfo(
        key="spf",
        name="SPF (Survey of Professional Forecasters)",
        family="Benchmark",
        reference="Faust–Wright (2013) — SPF",
        description=(
            "The Philadelphia Fed's Survey of Professional Forecasters. At each "
            "quarter's survey origin, ~40 professional forecasters submit "
            "point forecasts of the annualized CPI, PCE, GDP-deflator, and core "
            "CPI inflation rate at horizons 0..5 quarters. This wrapper reads the "
            "public Mean_Level.xlsx from Philly Fed and returns the mean forecast "
            "at the requested horizon. **This is the frontier** — FW find "
            "subjective forecasts dominate all model-based ones."
        ),
    )

    def _load(self, series_key: str):
        return load_spf(series_key)


class SurveyGreenbook(_SurveyBase):
    survey_name = "Greenbook"
    info = ModelInfo(
        key="gb",
        name="Greenbook (Fed staff forecast)",
        family="Benchmark",
        reference="Faust–Wright (2013) — Greenbook",
        description=(
            "The Fed Board's internal Greenbook (now Tealbook) forecast, released "
            "at every FOMC meeting. Provides forecasts at horizons 0..9 quarters "
            "for GDP-deflator, CPI, and core CPI inflation. Public releases are "
            "**embargoed 5 years**, so the file typically ends ~2019/20. "
            "Historically the highest-accuracy inflation forecast series."
        ),
    )

    def _load(self, series_key: str):
        # Greenbook is indexed by meeting date; keep only rows ≤ pi_last (base
        # class handles that). Retain the most recent Greenbook release per
        # quarter as the effective quarterly forecast.
        gb = load_greenbook(series_key)
        if gb is None:
            return None
        # Roll release dates to quarter start; keep the last release in each quarter.
        gb = gb.copy()
        gb.index = gb.index.to_period("Q").to_timestamp(how="start")
        gb = gb[~gb.index.duplicated(keep="last")]
        return gb


class SurveyBlueChip(_SurveyBase):
    survey_name = "Blue Chip surrogate"
    info = ModelInfo(
        key="bc",
        name="Blue Chip surrogate (MICH + EXPINF1YR)",
        family="Benchmark",
        reference="Faust–Wright (2013) — Blue Chip (surrogate)",
        description=(
            "Blue Chip Economic Indicators is subscription-only, so this wrapper "
            "ships a free surrogate: the mean of the Michigan Survey 1-year "
            "expected-inflation series (MICH) and the Cleveland Fed 1-year "
            "expected-inflation series (EXPINF1YR), both from FRED. This is a "
            "good-quality stand-in — FW note the three survey series are highly "
            "correlated with each other."
        ),
    )

    def _fit(self) -> None:
        self._series_key = _guess_series_key(self._y, self._X)
        s = load_blue_chip_surrogate()
        if s is None or s.empty:
            self._wide = None
            self._latest = None
            self._last_origin = None
            return
        pi_last = self._y.index[-1]
        s = s.loc[:pi_last]
        # Only one horizon (1-yr) is available; 4 quarters ≈ 1 year.
        self._wide = s.to_frame(name="h4")
        if self._wide.empty:
            self._latest = None
            self._last_origin = None
        else:
            self._latest = self._wide.iloc[-1]
            self._last_origin = self._wide.index[-1]

    def _forecast(self, h: int) -> float:
        if self._latest is None:
            return float(self._y.iloc[-1])
        return float(self._latest["h4"])
