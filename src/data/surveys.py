"""Loader for survey-based inflation forecasts (SPF, Greenbook) and a free surrogate
for Blue Chip that averages FRED-provided 1-year expected-inflation measures.

Data-file conventions
---------------------
User places files under ``data/surveys/`` in the repo root. The loader auto-detects
CSV vs Excel by extension and normalizes the sheet into a long DataFrame with
columns [origin, horizon, series, value].

* **SPF** — download ``Mean_Level.xlsx`` from
  https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters
  → Data Files → Level Mean Responses. Place at ``data/surveys/spf_mean_level.xlsx``.
  The sheet has columns ``YEAR, QUARTER, CPI1..CPI6, PCE1..PCE6, PGDP1..PGDP6,
  CORECPI1..CORECPI6``. Each ``<series><n>`` cell is the mean forecast of
  annualized inflation for horizon *n* quarters (n=1 = nowcast for the survey
  quarter; n=6 = 5 quarters out).

* **Greenbook** — download ``gbweb_row_format.xlsx`` from
  https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/greenbook-data-sets
  Place at ``data/surveys/greenbook_row_format.xlsx``. Look for columns like
  ``gPGDPF0..gPGDPF9``, ``gPCPIF0..gPCPIF9``, ``gPCCPIF0..gPCCPIF9`` (F0 = nowcast).
  Note: Greenbook data is embargoed 5 years so the file ends ~2019.

* **Blue Chip** — subscription only. We compute a free surrogate as the mean of
  Michigan 1-yr (MICH) and Cleveland Fed 1-yr (EXPINF1YR) from FRED. No file
  upload required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Package root = two levels up from this file
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SURVEYS_DIR = _PROJECT_ROOT / "data" / "surveys"


# --------------------------------------------------------------------------- #
# SPF loader
# --------------------------------------------------------------------------- #
_SPF_ALIASES = {
    "cpi": "CPI",
    "pce": "PCE",
    "gdpdef": "PGDP",
    "core_cpi": "CORECPI",
}


def _spf_path() -> Optional[Path]:
    """Find an SPF file if the user has placed one under data/surveys/."""
    for name in ("spf_mean_level.xlsx", "spf_mean_level.csv",
                 "Mean_Level.xlsx", "Mean_Level.csv"):
        p = SURVEYS_DIR / name
        if p.exists():
            return p
    return None


def load_spf(series: str = "cpi") -> Optional[pd.DataFrame]:
    """Return SPF quarterly mean forecasts for `series` as a wide DataFrame with
    index = survey origin (quarterly), columns = horizons h=0..5, values = the
    mean forecast of the annualized inflation rate at that horizon.

    Column names in the raw file use ``<PREFIX><n>`` where n = 1..6; SPF's own
    convention is n=1 → current quarter, n=6 → 5 quarters out. We remap to
    horizons h = 0..5 (so h=0 is the current-quarter nowcast).
    """
    path = _spf_path()
    if path is None:
        return None
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    prefix = _SPF_ALIASES.get(series.lower())
    if prefix is None:
        return None
    cols = [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    if not cols:
        return None
    # Build the origin index as quarter-start dates
    if "YEAR" in df.columns and "QUARTER" in df.columns:
        origin = pd.PeriodIndex.from_fields(
            year=df["YEAR"].astype(int),
            quarter=df["QUARTER"].astype(int),
            freq="Q",
        ).to_timestamp(how="start")
    else:
        return None
    out = pd.DataFrame(index=origin)
    for c in cols:
        n = int(c[len(prefix):])
        out[f"h{n-1}"] = pd.to_numeric(df[c], errors="coerce").values
    return out.sort_index()


# --------------------------------------------------------------------------- #
# Greenbook loader
# --------------------------------------------------------------------------- #
_GB_PREFIX = {
    "gdpdef": "gPGDP",
    "cpi":    "gPCPI",
    "core_cpi": "gPCCPI",
}


def _gb_path() -> Optional[Path]:
    for name in ("greenbook_row_format.xlsx", "greenbook_row_format.csv",
                 "gbweb_row_format.xlsx", "gbweb_row_format.csv"):
        p = SURVEYS_DIR / name
        if p.exists():
            return p
    return None


def load_greenbook(series: str = "cpi") -> Optional[pd.DataFrame]:
    """Return Greenbook forecasts for `series` as a wide DataFrame keyed by the
    date of the Greenbook release, with columns h=0..9.
    """
    path = _gb_path()
    if path is None:
        return None
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    prefix = _GB_PREFIX.get(series.lower())
    if prefix is None:
        return None
    # Column names look like gPCPIF0..gPCPIF9
    cols = [c for c in df.columns if c.startswith(prefix + "F") and c[len(prefix) + 1:].isdigit()]
    if not cols:
        return None
    # Origin date — try a few likely column names
    origin_col = None
    for cand in ("GBdate", "gbdate", "meeting_date", "MEETING_DATE", "date"):
        if cand in df.columns:
            origin_col = cand
            break
    if origin_col is None and {"GByear", "GBmonth"}.issubset(df.columns):
        origin = pd.to_datetime(dict(year=df["GByear"], month=df["GBmonth"], day=1))
    elif origin_col is not None:
        origin = pd.to_datetime(df[origin_col])
    else:
        return None

    out = pd.DataFrame(index=pd.DatetimeIndex(origin))
    for c in cols:
        h = int(c[len(prefix) + 1:])
        out[f"h{h}"] = pd.to_numeric(df[c], errors="coerce").values
    return out.sort_index()


# --------------------------------------------------------------------------- #
# Blue Chip surrogate
# --------------------------------------------------------------------------- #
def load_blue_chip_surrogate() -> Optional[pd.Series]:
    """Free stand-in for Blue Chip 1-yr inflation: average of MICH and EXPINF1YR."""
    from . import fred as _f
    if not _f.has_live_data():
        return None
    mich = _f._fetch_fred_series("MICH", "1978-01-01")
    exp1 = _f._fetch_fred_series("EXPINF1YR", "1982-01-01")
    if mich is None and exp1 is None:
        return None
    frames = [s for s in (mich, exp1) if s is not None]
    df = pd.concat(frames, axis=1)
    return df.mean(axis=1).dropna()


# --------------------------------------------------------------------------- #
# Introspection helpers for the UI
# --------------------------------------------------------------------------- #
@dataclass
class SurveyStatus:
    spf_present: bool
    spf_path: Optional[Path]
    gb_present: bool
    gb_path: Optional[Path]
    bc_available: bool

    def summary(self) -> str:
        parts = []
        parts.append(f"SPF: {'✅' if self.spf_present else '❌ (place file at data/surveys/spf_mean_level.xlsx)'}")
        parts.append(f"Greenbook: {'✅' if self.gb_present else '❌ (place file at data/surveys/greenbook_row_format.xlsx)'}")
        parts.append(f"Blue-Chip surrogate (MICH + EXPINF1YR): {'✅' if self.bc_available else '❌'}")
        return "  |  ".join(parts)


def survey_status() -> SurveyStatus:
    spf = _spf_path()
    gb = _gb_path()
    bc = load_blue_chip_surrogate()
    return SurveyStatus(
        spf_present=spf is not None, spf_path=spf,
        gb_present=gb is not None, gb_path=gb,
        bc_available=bc is not None,
    )
