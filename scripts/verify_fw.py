"""Cross-check the dashboard's Faust-Wright models against Table 1.2 of the paper.

Runs the same horse race on GDP-deflator inflation over FW's own sample window
(1985Q1-2011Q4), then prints our rel-RMSPE next to the published values.

Flags:
    --short     Skip the ~2s/fit DSGE models and use a smaller horizon set.
    --alfred    Use vintage GDPDEF + UNRATE from ALFRED at each origin (real-time
                training data, matching FW's own construction). Falls back to
                current-vintage series for origins before ALFRED coverage
                (GDPDEF earliest vintage = 1991-12).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import fred as fred_mod
from src.evaluation.fw_horserace import run_fw_horserace
from src.models import registry


# Faust-Wright (2013) Table 1.2 (GDP deflator, 1985Q1-2011Q4).
# rows keyed to dashboard model keys, columns = quarterly horizons.
FW_TABLE_1_2 = pd.DataFrame(
    {
        0: [1.06, 1.06, 1.07, 1.19, 0.95, 0.98, 1.03, 1.04, 1.04, 1.07, 0.99,
            1.02, 1.00, 1.02, 1.06, 1.02, 1.00],
        1: [1.00, 1.02, 1.03, 1.17, 0.90, 0.96, 0.97, 1.02, 1.02, 1.12, 0.94,
            0.94, 0.91, 1.03, 1.02, 0.95, 1.00],
        2: [0.96, 1.01, 1.01, 1.09, 0.91, 0.91, 0.95, 1.03, 1.03, 1.16, 0.95,
            0.91, 0.89, 1.07, 1.06, 0.97, 1.00],
        3: [1.04, 1.17, 1.08, 1.04, 0.94, 0.91, 1.01, 1.10, 1.10, 1.25, 0.94,
            0.97, 0.97, 1.06, 1.08, 0.98, 1.00],
        4: [1.09, 1.24, 1.14, 1.06, 0.96, 0.94, 1.05, 1.17, 1.17, 1.32, 1.00,
            1.01, 1.09, 1.13, 1.08, 0.97, 1.00],
        8: [1.34, 1.53, 1.41, 1.25, 1.05, 1.07, 1.18, 1.33, 1.30, 1.50, 1.21,
            1.15, 1.19, 1.26, 1.16, 1.05, 1.00],
    },
    index=["fw_direct", "fw_rar", "fw_pc", "rw", "ao", "ucsv",
           "fw_argap", "fw_pcgap", "fw_pctvngap", "fw_tsvar", "tvpvar",
           "fw_ewa", "fw_bma", "fw_favar", "sw07", "fw_dsgegap", "fw_fixedrho"],
)


def load_gdpdef_quarterly() -> tuple[pd.Series, pd.DataFrame | None]:
    """GDP deflator quarterly, annualized log-diff. Plus unemployment (quarterly)."""
    lvl = fred_mod._fetch_fred_series("GDPDEF", "1960-01-01")
    if lvl is None:
        raise SystemExit("Need a FRED key with GDPDEF access.")
    lvl_q = lvl.resample("QS").mean()
    pi = 400.0 * np.log(lvl_q / lvl_q.shift(1)).dropna()
    unrate = fred_mod._fetch_fred_series("UNRATE", "1960-01-01")
    if unrate is None:
        return pi, None
    u_q = unrate.resample("QS").mean()
    X = pd.DataFrame({"unrate": u_q})
    return pi, X


def _pi_from_level(lvl: pd.Series) -> pd.Series:
    """Quarterly annualized log-diff inflation from a monthly-or-quarterly level."""
    lvl_q = lvl.resample("QS").mean()
    return 400.0 * np.log(lvl_q / lvl_q.shift(1)).dropna()


def build_vintage_training(origins: pd.DatetimeIndex) -> dict:
    """For each origin, build the (y_train, X_train) pair from ALFRED as it was
    known at that origin date.

    Vintage for a Q_t origin = the latest release with realtime_start <= Q_t.
    If ALFRED has no coverage before ``origin`` (GDPDEF only starts 1991-12),
    ``fetch_vintage_series`` returns the current-vintage series clipped to that
    date instead — imperfect, but the best available approximation.
    """
    out = {}
    for origin in origins:
        # Vintage series known "on the morning of `origin`" — the model would
        # then forecast at horizon h >= 1 into future quarters.
        p_lvl = fred_mod.fetch_vintage_series("GDPDEF", origin)
        u_lvl = fred_mod.fetch_vintage_series("UNRATE", origin)
        if p_lvl is None or u_lvl is None:
            continue
        pi = _pi_from_level(p_lvl)
        # UNRATE is monthly; the horse race passes activity as-is (each model
        # aligns internally). Resample to Q to keep with pi's cadence.
        u_q = u_lvl.resample("QS").mean()
        # Clip both to strictly-before-origin so nothing peeks at future data.
        pi = pi[pi.index < origin]
        u_q = u_q[u_q.index < origin]
        X = pd.DataFrame({"unrate": u_q})
        out[pd.Timestamp(origin)] = (pi, X)
    return out


def main(short: bool = False, alfred: bool = False) -> None:
    pi, X = load_gdpdef_quarterly()
    # FW's evaluation sample: origins are 1985Q1..2011Q4 (inflation from 1960 onward)
    end = pd.Timestamp("2011-12-31")
    pi = pi.loc[:end]
    if X is not None:
        X = X.loc[:end]
    print(f"GDP deflator quarterly, {pi.index[0].date()} → {pi.index[-1].date()}, "
          f"n={len(pi)} obs")

    horizons = [0, 1, 4, 8] if short else [0, 1, 2, 3, 4, 8]
    keys = list(FW_TABLE_1_2.index)
    if short:
        # SW07 & DSGE-GAP are ~2s/fit; skip them in short mode
        keys = [k for k in keys if k not in ("sw07", "fw_dsgegap")]
    # min_train chosen so the first origin is roughly 1985Q1
    first_target = pd.Timestamp("1985-01-01")
    min_train = int((pi.index < first_target).sum())
    print(f"min_train={min_train} → first origin ≈ {pi.index[min_train].date()}")

    training_by_origin = None
    if alfred:
        origins = pi.index[min_train:len(pi) - max(horizons)]
        print(f"Building ALFRED vintage training set over {len(origins)} origins…")
        training_by_origin = build_vintage_training(origins)
        # Report ALFRED coverage
        vintage_hits = sum(
            1 for o in origins
            if o in training_by_origin
            and training_by_origin[o][0].index[-1] >= pd.Timestamp("1991-01-01")
        )
        print(f"  ALFRED-native origins (post-1991-12 vintages): {vintage_hits}/{len(origins)}")

    res = run_fw_horserace(
        pi, X, keys, horizons,
        benchmark_key="fw_fixedrho",
        min_train=min_train, step=1,
        n_workers=6,
        training_by_origin=training_by_origin,
    )

    ours = res.rel_rmspe.reindex(keys)
    theirs = FW_TABLE_1_2.loc[keys, horizons]

    print("\n=== Faust-Wright Table 1.2 (GDP deflator, 1985Q1-2011Q4) ===")
    print("model            " + "".join(f"{'h=' + str(h):>8s}" for h in horizons))
    print("-" * (17 + 8 * len(horizons)))
    for k in keys:
        name = registry.make(k).info.name[:16]
        row_o = "".join(f"{ours.at[k, h]:8.2f}" for h in horizons)
        row_t = "".join(f"{theirs.at[k, h]:8.2f}" for h in horizons)
        print(f"{name:16s} ours: {row_o}")
        print(f"{'':16s} FW:   {row_t}")
        diff = "".join(f"{ours.at[k, h] - theirs.at[k, h]:+8.2f}" for h in horizons)
        print(f"{'':16s} d :   {diff}")

    # rank correlation
    print("\n=== Rank correlation across models (per horizon) ===")
    for h in horizons:
        common = ours[h].dropna().index.intersection(theirs[h].dropna().index)
        corr = ours.loc[common, h].rank().corr(theirs.loc[common, h].rank(),
                                                method="pearson")
        print(f"  h={h}: Spearman ρ = {corr:.3f}  "
              f"(n={len(common)})")


if __name__ == "__main__":
    main(short="--short" in sys.argv, alfred="--alfred" in sys.argv)
