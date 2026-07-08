"""Pass-through forecast models for the three Faust–Wright survey benchmarks.

Faust–Wright's most important empirical finding is that **judgmental survey
forecasts dominate all model-based forecasts** — often by a wide margin. These
three wrappers plug the surveys into the same ForecastModel interface as
everything else, so they show up in the horse race on equal footing.

* SurveySPF        — Survey of Professional Forecasters (Philly Fed).
* SurveyGreenbook  — Fed Board Greenbook / Tealbook (Philly Fed archive).
* SurveyBlueChip   — Blue Chip Consensus. Since Blue Chip is subscription-only
                     we ship a free surrogate: the mean of Michigan and
                     Cleveland Fed 1-yr expected-inflation series from FRED.
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


def _series_key_for(pi_index_name: str | None) -> str:
    # Not currently used — the horse race passes the inflation-measure key into
    # the wrapper via a class-level attribute set by the tab. This shim is here
    # in case we want to auto-detect from the Series name in future.
    return "cpi"


class _SurveyBase(ForecastModel):
    """Common lookup logic: pick the most recent survey origin ≤ the last fit-time
    date, then read the horizon-h column."""

    survey_name: str = "Survey"
    series_key: str = "cpi"                  # can be overridden at construction
    _wide: Optional[pd.DataFrame] = None

    def __init__(self, series_key: str = "cpi"):
        super().__init__(series_key=series_key)
        self.series_key = series_key

    def _load(self) -> Optional[pd.DataFrame]:
        raise NotImplementedError

    def _fit(self) -> None:
        w = self._load()
        if w is None or w.empty:
            self._wide = None
            self._latest = None
            self._last_origin = None
            return
        # Keep only rows whose origin is on or before the training end
        pi_last = self._y.index[-1]
        w = w.loc[:pi_last]
        self._wide = w
        self._latest = w.iloc[-1] if not w.empty else None
        self._last_origin = w.index[-1] if not w.empty else None

    def _forecast(self, h: int) -> float:
        if self._wide is None or self._latest is None:
            # Fall back to the last inflation observation
            return float(self._y.iloc[-1])
        # Survey horizons are labeled h0, h1, ... Clip requested h into available.
        cols = list(self._latest.index)
        col = f"h{h}"
        if col not in cols:
            # Fall back to the closest available horizon (usually the largest one)
            available = sorted(int(c[1:]) for c in cols if c.startswith("h"))
            if not available:
                return float(self._y.iloc[-1])
            h_use = max(x for x in available if x <= h) if any(x <= h for x in available) else available[-1]
            col = f"h{h_use}"
        val = self._latest.get(col, np.nan)
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

    def _load(self) -> Optional[pd.DataFrame]:
        return load_spf(self.series_key)


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
            "**embargoed 5 years**, so the file typically ends ~2019. Historically "
            "the highest-accuracy inflation forecast series."
        ),
    )

    def _load(self) -> Optional[pd.DataFrame]:
        return load_greenbook(self.series_key)


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
        s = load_blue_chip_surrogate()
        if s is None or s.empty:
            self._wide = None
            self._latest = None
            self._last_origin = None
            return
        pi_last = self._y.index[-1]
        s = s.loc[:pi_last]
        # Only one horizon (1-yr) is available for the surrogate.
        self._wide = s.to_frame(name="h4")   # 1yr ≈ 4 quarters
        self._latest = self._wide.iloc[-1]
        self._last_origin = self._wide.index[-1]

    def _forecast(self, h: int) -> float:
        if self._latest is None:
            return float(self._y.iloc[-1])
        return float(self._latest["h4"])
