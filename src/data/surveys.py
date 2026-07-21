"""Loader for survey-based inflation forecasts (SPF, Greenbook) and a free surrogate
for Blue Chip.

File-format details (learned from actual Philly Fed downloads)
--------------------------------------------------------------
* Both files come from a SAS export pipeline that writes a **malformed
  ``docProps/core.xml``** timestamp (a space where the hour digit should be
  padded), which crashes openpyxl. We rewrite the XML in memory before parsing
  — never touch the file on disk.

* **SPF Mean_Level.xlsx** — one *sheet per series*. The inflation sheets are
  ``CPI``, ``CORECPI``, ``PCE``, ``COREPCE``, ``PGDP`` (GDP deflator). Each sheet
  is wide: ``YEAR, QUARTER, <SERIES>1..<SERIES>6``. Column ``<SERIES>1`` is the
  survey forecast for the *current* quarter (i.e. nowcast, h=0); ``<SERIES>6``
  is 5 quarters out (h=5). Values are **annualized quarterly inflation in
  percent**.

* **Greenbook row_format.xlsx** — one *sheet per series*. Inflation sheets:
  ``gPCPI`` (headline CPI), ``gPCPIX`` (core CPI), ``gPPCE`` (headline PCE),
  ``gPPCEX`` (core PCE), ``gPGDP`` (GDP deflator). Layout: ``DATE,
  <SERIES>B4..B1, <SERIES>F0..F9, GBdate``. ``DATE`` is a *decimal* quarter
  code (e.g. 2020.2 = 2020 Q2) — the reference quarter that ``F0`` describes.
  ``F0..F9`` = nowcast + 9-quarter forecast. ``B*`` are backcasts (not used).
  ``GBdate`` is the meeting release date as ``YYYYMMDD``. Values are already in
  the model's units (annualized quarterly log-difference, percent).
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SURVEYS_DIR = _PROJECT_ROOT / "data" / "surveys"


# --------------------------------------------------------------------------- #
# xlsx reader that tolerates the malformed core.xml from Philly Fed's SAS pipe
# --------------------------------------------------------------------------- #
def _open_xlsx_forgiving(path: Path) -> pd.ExcelFile:
    """Return an ExcelFile for ``path`` even if its ``docProps/core.xml`` has a
    malformed timestamp (missing hour zero-padding, common in Philly Fed's SAS
    exports). We rewrite the metadata in an in-memory zip copy and hand that
    buffer to pandas — the source file on disk is untouched.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            data = zin.read(it.filename)
            if it.filename == "docProps/core.xml":
                # "T 2:56:52-04:00"  ->  "T02:56:52-04:00"
                data = re.sub(rb"T\s+(\d:)", rb"T0\1", data)
            zout.writestr(it, data)
    buf.seek(0)
    return pd.ExcelFile(buf, engine="openpyxl")


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def _spf_path() -> Optional[Path]:
    for name in ("spf_mean_level.xlsx", "Mean_Level.xlsx",
                 "spf_mean_level.xls",  "Mean_Level.xls"):
        p = SURVEYS_DIR / name
        if p.exists():
            return p
    return None


def _gb_path() -> Optional[Path]:
    for name in ("greenbook_row_format.xlsx", "gbweb_row_format.xlsx",
                 "greenbook_row_format.xls", "gbweb_row_format.xls"):
        p = SURVEYS_DIR / name
        if p.exists():
            return p
    return None


def _bc_lr_path() -> Optional[Path]:
    for name in ("blue_chip_lr.xlsx", "bcei_lr.xlsx",
                 "blue_chip_lr.xls",  "bcei_lr.xls"):
        p = SURVEYS_DIR / name
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------- #
# SPF loader
# --------------------------------------------------------------------------- #
# key -> (sheet_name, column_prefix)
_SPF_SHEET = {
    "cpi":       ("CPI",       "CPI"),
    "core_cpi":  ("CORECPI",   "CORECPI"),
    "pce":       ("PCE",       "PCE"),
    "core_pce":  ("COREPCE",   "COREPCE"),
    "gdpdef":    ("PGDP",      "PGDP"),
}


@lru_cache(maxsize=2)
def load_spf_cpi10() -> Optional[pd.Series]:
    """SPF 10-year-ahead CPI inflation forecast (annualized %).

    Quarterly consensus mean from the Philly Fed SPF (sheet ``CPI10``, one column
    of the same name). Available since 1991Q4. Serves as a Blue-Chip 5–10y CPI
    stand-in for Faust–Wright's local-mean trend τ_t — same object (a
    professional-forecaster long-horizon anchor), higher data quality than the
    5-yr Michigan/EXPINF surrogate, and free.

    Returned as a Series indexed by the survey origin quarter (quarter start).
    """
    path = _spf_path()
    if path is None:
        return None
    try:
        xf = _open_xlsx_forgiving(path)
        if "CPI10" not in xf.sheet_names:
            return None
        df = xf.parse("CPI10")
    except Exception:
        return None
    if not {"YEAR", "QUARTER", "CPI10"}.issubset(df.columns):
        return None
    df = df.dropna(subset=["CPI10"])
    origin = pd.PeriodIndex.from_fields(
        year=df["YEAR"].astype(int),
        quarter=df["QUARTER"].astype(int),
        freq="Q",
    ).to_timestamp(how="start")
    return pd.Series(pd.to_numeric(df["CPI10"], errors="coerce").values,
                     index=origin, name="spf_cpi10").sort_index().dropna()


@lru_cache(maxsize=8)
def load_spf(series: str = "cpi") -> Optional[pd.DataFrame]:
    """Return SPF quarterly mean forecasts for ``series`` as a wide DataFrame
    indexed by the *survey origin* (the quarter the survey was conducted),
    columns = horizons ``h0..h5``.

    The SPF numbering has ``<PREFIX>1`` = forecast for the survey quarter
    itself (h=0), ``<PREFIX>2`` = h=1 quarters ahead, ..., ``<PREFIX>6`` = h=5.
    """
    path = _spf_path()
    if path is None:
        return None
    spec = _SPF_SHEET.get(series.lower())
    if spec is None:
        return None
    sheet, prefix = spec
    try:
        xf = _open_xlsx_forgiving(path)
        if sheet not in xf.sheet_names:
            return None
        df = xf.parse(sheet)
    except Exception:
        return None
    if not {"YEAR", "QUARTER"}.issubset(df.columns):
        return None
    origin = pd.PeriodIndex.from_fields(
        year=df["YEAR"].astype(int),
        quarter=df["QUARTER"].astype(int),
        freq="Q",
    ).to_timestamp(how="start")
    out = pd.DataFrame(index=origin)
    for n in range(1, 7):
        col = f"{prefix}{n}"
        if col in df.columns:
            out[f"h{n-1}"] = pd.to_numeric(df[col], errors="coerce").values
    return out.sort_index().dropna(how="all")


# --------------------------------------------------------------------------- #
# Greenbook loader
# --------------------------------------------------------------------------- #
# key -> (sheet_name, series_prefix)
_GB_SHEET = {
    "cpi":       ("gPCPI",  "gPCPI"),
    "core_cpi":  ("gPCPIX", "gPCPIX"),
    "pce":       ("gPPCE",  "gPPCE"),
    "core_pce":  ("gPPCEX", "gPPCEX"),
    "gdpdef":    ("gPGDP",  "gPGDP"),
}


@lru_cache(maxsize=8)
def load_greenbook(series: str = "cpi") -> Optional[pd.DataFrame]:
    """Return Greenbook forecasts for ``series`` as a wide DataFrame indexed by
    the *Greenbook release date* (from ``GBdate`` = YYYYMMDD), columns = h0..h9.
    """
    path = _gb_path()
    if path is None:
        return None
    spec = _GB_SHEET.get(series.lower())
    if spec is None:
        return None
    sheet, prefix = spec
    try:
        xf = _open_xlsx_forgiving(path)
        if sheet not in xf.sheet_names:
            return None
        df = xf.parse(sheet)
    except Exception:
        return None
    if "GBdate" not in df.columns:
        return None
    origin = pd.to_datetime(df["GBdate"].astype(int).astype(str), format="%Y%m%d",
                            errors="coerce")
    out = pd.DataFrame(index=origin)
    for h in range(10):
        col = f"{prefix}F{h}"
        if col in df.columns:
            out[f"h{h}"] = pd.to_numeric(df[col], errors="coerce").values
    out = out.dropna(subset=[c for c in out.columns if c.startswith("h")],
                     how="all").sort_index()
    # Keep the *last* Greenbook per calendar quarter so downstream logic works
    # against a quarterly index. Reindex origin to the release quarter.
    return out


# --------------------------------------------------------------------------- #
# Blue Chip long-range (5-10y) — τ anchor used by Faust-Wright's Table 1.2
# --------------------------------------------------------------------------- #
# Expected file:  data/surveys/blue_chip_lr.xlsx
# Expected shape (one sheet per measure, sheet names case-insensitive):
#   sheet "GDPDEF" (or "PGDP"): columns YEAR, QUARTER, VALUE   (percent, annualized)
#   sheet "CPI":                columns YEAR, QUARTER, VALUE
#   sheet "PCE":                columns YEAR, QUARTER, VALUE
# Any subset of measures may be present. Values are annualized percent.
#
# Falls back gracefully: if a measure's sheet is missing, load_blue_chip_lr
# returns None for that measure and the τ-builder layers in the next-best
# anchor (SPF CPI10, EXPINF10YR, or exp-smoothing).
_BC_LR_SHEET_ALIASES = {
    "gdpdef": ("gdpdef", "pgdp", "gdp_deflator", "deflator"),
    "cpi":    ("cpi", "cpi_headline", "headline_cpi"),
    "pce":    ("pce", "pce_headline", "headline_pce"),
}


@lru_cache(maxsize=8)
def load_blue_chip_lr(measure: str = "gdpdef") -> Optional[pd.Series]:
    """Blue Chip Economic Indicators 5-10y long-range inflation forecast.

    Returns a quarterly Series indexed by survey origin (quarter start), values
    in annualized percent. This is FW's actual τ anchor for Table 1.2 (Blue
    Chip GDPDEF LR); the SPF CPI10 / EXPINF10YR paths in local_mean_trend are
    open-source stand-ins used only when this file is absent.

    ``measure`` in {"gdpdef", "cpi", "pce"} — case-insensitive.
    """
    path = _bc_lr_path()
    if path is None:
        return None
    aliases = _BC_LR_SHEET_ALIASES.get(measure.lower())
    if not aliases:
        return None
    try:
        xf = _open_xlsx_forgiving(path)
    except Exception:
        return None
    sheet_map = {s.lower(): s for s in xf.sheet_names}
    sheet = next((sheet_map[a] for a in aliases if a in sheet_map), None)
    if sheet is None:
        return None
    try:
        df = xf.parse(sheet)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    # Accept either (YEAR, QUARTER, VALUE) or (DATE, VALUE) layout.
    if {"year", "quarter"}.issubset(cols):
        val_col = next((cols[k] for k in ("value", "bc_lr", "bcei_lr", "lr")
                        if k in cols), None)
        if val_col is None:
            # single non-index numeric column
            num_cols = [c for c in df.columns
                        if c not in (cols["year"], cols["quarter"])
                        and pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) != 1:
                return None
            val_col = num_cols[0]
        df = df.dropna(subset=[cols["year"], cols["quarter"], val_col])
        origin = pd.PeriodIndex.from_fields(
            year=df[cols["year"]].astype(int),
            quarter=df[cols["quarter"]].astype(int),
            freq="Q",
        ).to_timestamp(how="start")
        vals = pd.to_numeric(df[val_col], errors="coerce").values
    elif "date" in cols:
        val_col = next((cols[k] for k in ("value", "bc_lr", "bcei_lr", "lr")
                        if k in cols), None)
        if val_col is None:
            num_cols = [c for c in df.columns
                        if c != cols["date"]
                        and pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) != 1:
                return None
            val_col = num_cols[0]
        df = df.dropna(subset=[cols["date"], val_col])
        origin = pd.to_datetime(df[cols["date"]], errors="coerce")
        vals = pd.to_numeric(df[val_col], errors="coerce").values
    else:
        return None
    s = pd.Series(vals, index=origin, name=f"bc_lr_{measure}").sort_index().dropna()
    # Snap origins to quarter start for clean alignment downstream.
    s.index = s.index.to_period("Q").to_timestamp(how="start")
    s = s[~s.index.duplicated(keep="last")]
    return s


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
    frames = [s for s in (mich, exp1) if s is not None]
    if not frames:
        return None
    return pd.concat(frames, axis=1).mean(axis=1).dropna()


# --------------------------------------------------------------------------- #
# Status object
# --------------------------------------------------------------------------- #
@dataclass
class SurveyStatus:
    spf_present: bool
    spf_path: Optional[Path]
    gb_present: bool
    gb_path: Optional[Path]
    bc_available: bool
    bc_lr_present: bool
    bc_lr_path: Optional[Path]
    bc_lr_measures: tuple[str, ...]

    def summary(self) -> str:
        parts = []
        parts.append(f"**SPF**: {'✅ ' + self.spf_path.name if self.spf_present else '❌ (place at data/surveys/spf_mean_level.xlsx)'}")
        parts.append(f"**Greenbook**: {'✅ ' + self.gb_path.name if self.gb_present else '❌ (place at data/surveys/greenbook_row_format.xlsx)'}")
        parts.append(f"**Blue-Chip surrogate** (MICH + EXPINF1YR): {'✅' if self.bc_available else '❌'}")
        if self.bc_lr_present:
            meas = ", ".join(self.bc_lr_measures) if self.bc_lr_measures else "none"
            parts.append(f"**Blue-Chip 5–10y** (τ anchor): ✅ {self.bc_lr_path.name}  ({meas})")
        else:
            parts.append("**Blue-Chip 5–10y** (τ anchor): ❌ (place at data/surveys/blue_chip_lr.xlsx)")
        return "  |  ".join(parts)


def survey_status() -> SurveyStatus:
    spf = _spf_path()
    gb = _gb_path()
    bc = load_blue_chip_surrogate()
    bc_lr = _bc_lr_path()
    measures = tuple(m for m in ("gdpdef", "cpi", "pce")
                     if load_blue_chip_lr(m) is not None) if bc_lr else ()
    return SurveyStatus(
        spf_present=spf is not None, spf_path=spf,
        gb_present=gb is not None, gb_path=gb,
        bc_available=bc is not None,
        bc_lr_present=bc_lr is not None, bc_lr_path=bc_lr,
        bc_lr_measures=measures,
    )
