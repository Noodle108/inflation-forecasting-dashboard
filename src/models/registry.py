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
from .bvar import BVAR, BVARHierarchical
from .dsge import NewKeynesianPC, SmallScaleDSGE
from .phillips import PhillipsCurve
from .sw2007 import SmetsWouters2007
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
    "nkpc": NewKeynesianPC,
    "dsge": SmallScaleDSGE,
    "sw07": SmetsWouters2007,
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
