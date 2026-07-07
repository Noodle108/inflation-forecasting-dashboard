"""Common interface every forecasting model implements.

The whole dashboard and the evaluation harness only ever talk to this interface,
so adding a new model (statistical or structural) is purely a matter of subclassing
`ForecastModel` and registering it.

A model is *fit* on a history of inflation (plus optional exogenous activity data),
then asked to produce an h-step-ahead forecast. The convention throughout is the
**h-step direct forecast of average inflation** over the next h periods is left to
the caller; `forecast(h)` returns the period-t+h point forecast. Multi-horizon
comparison in the backtester iterates over h.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModelInfo:
    key: str
    name: str
    family: str             # 'Benchmark' | 'Statistical' | 'Structural'
    reference: str          # short citation
    description: str        # one-paragraph plain-English explanation
    needs_activity: bool = False
    implemented: bool = True
    # --- richer, comparison-oriented context (shown next to the chart) ---
    citation: str = ""      # fuller reference / the paper it comes from
    intuition: str = ""     # how it forms a forecast, in plain English
    unique: str = ""        # what distinguishes it from the other models here
    strengths: str = ""     # when/why it tends to do well
    caveats: str = ""       # known weaknesses / what to watch for
    forecast_shape: str = ""  # what its forecast *path* looks like and why
    # --- data + methodology, shown in Model Library and expander cards ---
    data_sources: List[Tuple[str, str]] = field(default_factory=list)  # [(label, url)]
    assumptions: str = ""       # key modeling assumptions (Fisher, real wages, RE, ...)
    equations: str = ""         # a compact math sketch of what the model actually is


class ForecastModel(ABC):
    """Base class. Subclasses set `info` and implement `fit` + `_forecast`."""

    info: ModelInfo

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self._y: Optional[pd.Series] = None
        self._X: Optional[pd.DataFrame] = None
        self._fitted = False

    # -- lifecycle -----------------------------------------------------------
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "ForecastModel":
        """Fit on inflation history `y` (and optional activity frame `X`)."""
        self._y = y.astype(float).dropna()
        # align activity to the inflation dates by *label* (robust to the one-period
        # offset from differencing and to any ragged edges); trailing ffill only.
        self._X = X.reindex(self._y.index).ffill() if X is not None else None
        self._fit()
        self._fitted = True
        return self

    def forecast(self, h: int) -> float:
        """Point forecast of inflation `h` periods after the last fitted obs."""
        if not self._fitted:
            raise RuntimeError(f"{self.info.key}: call fit() before forecast().")
        if h < 1:
            raise ValueError("h must be >= 1")
        return float(self._forecast(h))

    # -- to implement --------------------------------------------------------
    @abstractmethod
    def _fit(self) -> None:
        ...

    @abstractmethod
    def _forecast(self, h: int) -> float:
        ...

    # -- convenience ---------------------------------------------------------
    def __repr__(self) -> str:
        return f"<{self.info.key} hyperparams={self.hyperparams}>"
