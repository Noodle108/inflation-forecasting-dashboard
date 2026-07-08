"""Central model registry.

The dashboard and the evaluation harness iterate over `MODELS`. To add a model,
implement it against `ForecastModel` and add a factory entry here — it then appears
in the UI and the backtester automatically.
"""
from __future__ import annotations

from typing import Callable, Dict

from .ar import ARModel
from .base import ForecastModel
from .benchmarks import AtkesonOhanian, RandomWalk
from .bernanke_blanchard import BernankeBlanchard
from .bvar import BVAR, BVARHierarchical
from .cleveland_expectations import ClevelandExpectations
from .cleveland_nowcast import ClevelandNowcast
from .dsge import NewKeynesianPC, SmallScaleDSGE
from .faust_wright import (
    ARGap, BMALargeDS, DSGEGap, DirectAR, EWALargeDS, FAVARLargeDS,
    FixedRhoGap, PCGap, PCTVNGap, PhillipsFW, RecursiveAR, TermStructureVAR,
)
from .nyfed_dsge import NYFedDSGE
from .phillips import PhillipsCurve
from .surveys import SurveyBlueChip, SurveyGreenbook, SurveySPF
from .sw2007 import SmetsWouters2007
from .sw_dfm import StockWatsonDFM
from .tvt_nkpc import TVTNKPC
from .tvp_var import TVPVARSV
from .ucsv import UCSV, UCSVSV

# key -> zero-arg factory returning a fresh, unfit model instance
MODELS: Dict[str, Callable[[], ForecastModel]] = {
    "rw": RandomWalk,
    "ao": AtkesonOhanian,
    "ar": ARModel,
    "pc": PhillipsCurve,
    "ucsv": UCSV,
    "ucsvsv": UCSVSV,
    "bvar": BVAR,
    "bvarh": BVARHierarchical,
    "tvpvar": TVPVARSV,
    "nkpc": NewKeynesianPC,
    "tvtnkpc": TVTNKPC,
    "swdfm": StockWatsonDFM,
    "clenow": ClevelandNowcast,
    "dsge": SmallScaleDSGE,
    "sw07": SmetsWouters2007,
    "bb": BernankeBlanchard,
    "nyfed": NYFedDSGE,
    "cleexp": ClevelandExpectations,
    # ----- Faust–Wright (2013) horse-race models -----
    "fw_direct": DirectAR,
    "fw_rar": RecursiveAR,
    "fw_pc": PhillipsFW,
    "fw_argap": ARGap,
    "fw_fixedrho": FixedRhoGap,
    "fw_pcgap": PCGap,
    "fw_pctvngap": PCTVNGap,
    "fw_tsvar": TermStructureVAR,
    "fw_ewa": EWALargeDS,
    "fw_bma": BMALargeDS,
    "fw_favar": FAVARLargeDS,
    "fw_dsgegap": DSGEGap,
    # ----- Survey benchmarks (FW's frontier) -----
    "spf": SurveySPF,
    "gb": SurveyGreenbook,
    "bc": SurveyBlueChip,
}

# Which model keys correspond to which line in Faust–Wright's Table 1.2. Reused
# by the Faust–Wright tab to build the RMSPE leaderboard. Order matches Table 1.2.
FW_TABLE_KEYS: list[str] = [
    "fw_direct",   # Direct
    "fw_rar",      # RAR
    "fw_pc",       # PC
    "rw",          # RW (already in the app)
    "ao",          # RW-AO (already in the app; window=12 monthly / 4 quarterly)
    "ucsv",        # UCSV (constant-vol approximation; app also has ucsvsv)
    "fw_argap",    # AR-GAP
    "fw_pcgap",    # PC-GAP
    "fw_pctvngap", # PCTVN-GAP
    "fw_tsvar",    # Term Structure VAR
    "tvpvar",      # TVP-VAR (already in the app)
    "fw_ewa",      # EWA
    "fw_bma",      # BMA
    "fw_favar",    # FAVAR
    "sw07",        # DSGE (SW07)
    "fw_dsgegap",  # DSGE-GAP
    # --- Subjective survey forecasts (FW's frontier) ---
    "bc",          # Blue Chip (surrogate: MICH + EXPINF1YR average)
    "spf",         # SPF mean forecast (requires data/surveys/spf_mean_level.xlsx)
    "gb",          # Greenbook (requires data/surveys/greenbook_row_format.xlsx)
    "fw_fixedrho", # Fixed ρ  (benchmark — divisor for the relative RMSPE column)
]
FW_BENCHMARK_KEY = "fw_fixedrho"

# Default benchmark all skill scores are computed against.
BENCHMARK_KEY = "rw"


# Merge per-model data-source / assumptions / equations metadata (see model_extras.py).
from .model_extras import apply_extras as _apply_extras
_apply_extras(MODELS)


def make(key: str) -> ForecastModel:
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Available: {list(MODELS)}")
    return MODELS[key]()


def all_infos():
    """List of ModelInfo for every registered model."""
    return [make(k).info for k in MODELS]
