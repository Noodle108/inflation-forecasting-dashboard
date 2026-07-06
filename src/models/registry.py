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
from .nyfed_dsge import NYFedDSGE
from .phillips import PhillipsCurve
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
}

# Default benchmark all skill scores are computed against.
BENCHMARK_KEY = "rw"


def make(key: str) -> ForecastModel:
    if key not in MODELS:
        raise KeyError(f"Unknown model '{key}'. Available: {list(MODELS)}")
    return MODELS[key]()


def all_infos():
    """List of ModelInfo for every registered model."""
    return [make(k).info for k in MODELS]
