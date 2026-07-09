"""Inflation Forecasting Dashboard — Streamlit front end.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# make `src` importable when run via `streamlit run`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# On Streamlit Community Cloud there is no .env; the FRED key lives in st.secrets.
# Bridge it into the environment before the data layer reads it.
if not os.environ.get("FRED_API_KEY"):
    try:
        if "FRED_API_KEY" in st.secrets:
            os.environ["FRED_API_KEY"] = str(st.secrets["FRED_API_KEY"])
    except Exception:
        pass

from src.data import fred
from src.evaluation.backtest import run_backtest
from src.evaluation.fw_horserace import run_fw_horserace
from src.models import registry
from src.models.model_extras import EXTRAS as _MODEL_EXTRAS

# Import the FW constants defensively: on Streamlit Cloud a stale-bytecode of
# registry.py from an earlier deploy can lack these names. If the import fails,
# fall back to the current values so the FW tab still renders.
try:
    from src.models.registry import FW_BENCHMARK_KEY, FW_TABLE_KEYS
except ImportError:
    FW_BENCHMARK_KEY = "fw_fixedrho"
    FW_TABLE_KEYS = [
        "fw_direct", "fw_rar", "fw_pc", "rw", "ao", "ucsv",
        "fw_argap", "fw_pcgap", "fw_pctvngap", "fw_tsvar", "tvpvar",
        "fw_ewa", "fw_bma", "fw_favar", "sw07", "fw_dsgegap", "fw_fixedrho",
    ]


def model_extras(key: str) -> dict:
    """Return the (data_sources, assumptions, equations) trio for a model key,
    reading from the extras dict directly. Robust to a stale cached ModelInfo
    class on Streamlit Cloud (which may lack these fields)."""
    return _MODEL_EXTRAS.get(key) or {}

st.set_page_config(page_title="Inflation Forecasting Dashboard", layout="wide",
                   page_icon="📈")

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
def _is_dark_theme() -> bool:
    """Best-effort read of Streamlit's active theme. Falls back to light."""
    try:
        base = st.get_option("theme.base")
        if base:
            return str(base).lower() == "dark"
    except Exception:
        pass
    # Newer Streamlit exposes context.theme.type
    try:
        return getattr(st.context.theme, "type", "light").lower() == "dark"
    except Exception:
        return False


DARK = _is_dark_theme()

# Theme-aware ink / grid / band, but keep the *forecast* line colors constant so
# the same model reads the same on either theme.
COLORS = {
    "ink":      "#e5e7eb" if DARK else "#0f172a",
    "muted":    "#94a3b8" if DARK else "#64748b",
    "grid":     "rgba(148,163,184,0.25)" if DARK else "#e2e8f0",
    "band":     "rgba(148,163,184,0.10)" if DARK else "rgba(15,23,42,0.04)",
    "realized": "#f8fafc" if DARK else "#0f172a",
    # Family colors used in the tables/captions
    "Benchmark":   "#94a3b8",
    "Statistical": "#60a5fa" if DARK else "#2563eb",
    "Structural":  "#f87171" if DARK else "#dc2626",
}
# Distinct colors within a family so multiple models can be told apart on the chart.
PALETTE = {
    "Benchmark":   ["#94a3b8", "#64748b", "#475569"],
    "Statistical": ["#2563eb", "#0ea5e9", "#7c3aed", "#0891b2", "#059669",
                    "#4f46e5", "#0369a1", "#1d4ed8", "#4338ca"],
    "Structural":  ["#dc2626", "#ea580c", "#b91c1c", "#c2410c", "#9f1239",
                    "#e11d48", "#f97316"],
}


def color_for(key: str, family: str, chosen: list[str]) -> str:
    """Assign each chosen model a stable color within its family palette."""
    same_family = [k for k in chosen if infos[k].family == family]
    idx = same_family.index(key) if key in same_family else 0
    pal = PALETTE.get(family, [COLORS["muted"]])
    return pal[idx % len(pal)]


# Global Plotly template — clean, minimal, theme-aware. Transparent backgrounds
# and theme-driven text/grid colors let the chart blend into either light or
# dark Streamlit themes without any hard-coded white panels.
CHART_LAYOUT = dict(
    template="plotly_dark" if DARK else "simple_white",
    font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif",
              size=13, color=COLORS["ink"]),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(30,41,59,0.95)" if DARK else "rgba(255,255,255,0.95)",
        bordercolor=COLORS["grid"],
        font=dict(family="Inter, system-ui, sans-serif", size=12,
                  color=COLORS["ink"]),
    ),
    xaxis=dict(showgrid=False, showline=True, linecolor=COLORS["grid"],
               ticks="outside", tickcolor=COLORS["grid"],
               tickfont=dict(size=12, color=COLORS["ink"])),
    yaxis=dict(gridcolor=COLORS["grid"], zeroline=False, showline=False,
               tickfont=dict(size=12, color=COLORS["ink"]),
               ticksuffix="%"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=COLORS["ink"])),
    margin=dict(t=50, b=40, l=10, r=20),
)


# --------------------------------------------------------------------------- #
# Cached data + model fits
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading macro data…")
def get_data(freq: str):
    return fred.load_data(freq=freq)


@st.cache_resource(show_spinner=False)
def fit_model(freq: str, infl_key: str, key: str, hp_json: str = ""):
    """Fit and cache a single model.

    ``hp_json`` — JSON-serialized hyperparameter dict. Passed positionally so
    Streamlit's cache keys on the actual values (dicts aren't hashable). Empty
    string means "use the class defaults" and is the fast-path.
    """
    d = get_data(freq)
    y = d.series(infl_key)
    X = d.activity if not d.activity.empty else None
    hp = json.loads(hp_json) if hp_json else {}
    # instantiate with custom hyperparams if any, else default factory
    if hp:
        cls = type(registry.make(key))    # get the class
        m = cls(**hp)
    else:
        m = registry.make(key)
    m.fit(y, X)
    return m


def forecast_path(model, H: int) -> np.ndarray:
    return np.array([model.forecast(h) for h in range(1, H + 1)])


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("📈 Inflation Forecasts")
st.sidebar.caption("Canonical inflation models on live FRED data.")

freq = st.sidebar.radio("Frequency", ["M", "Q"],
                        format_func={"M": "Monthly", "Q": "Quarterly"}.get)
data = get_data(freq)

if not data.is_synthetic:
    st.sidebar.success("Live FRED data connected.")
elif fred.has_live_data():
    st.sidebar.error(
        "FRED key found but the data request failed (network/rate limit) — "
        "showing a **synthetic** demo series. Try reloading."
    )
else:
    st.sidebar.warning(
        "No FRED_API_KEY found — using a **synthetic** demo series. "
        "Add a key in `.env` (local) or `st.secrets` (cloud) for live data."
    )

labels = data.price_labels
infl_key = st.sidebar.selectbox(
    "Inflation measure", list(data.inflation.columns),
    format_func=lambda k: labels.get(k, k),
)

infos = {i.key: i for i in registry.all_infos()}

# Group models by family for the sidebar picker. Order the families the way
# we visualize them (grey → blue → red) and sort within each family so the
# most-canonical models appear first.
_FAMILY_ORDER = ["Benchmark", "Statistical", "Structural"]
# Color-coded circles matching the chart palette: grey / blue / red.
_FAMILY_EMOJI = {"Benchmark": "⚪", "Statistical": "🔵", "Structural": "🔴"}
_by_family: dict[str, list[str]] = {f: [] for f in _FAMILY_ORDER}
for k, i in infos.items():
    _by_family.setdefault(i.family, []).append(k)
# Hide FW-only variants and survey benchmarks from the main sidebar (they're
# accessed via the Faust–Wright tab instead) to keep this list focused.
_HIDE_FROM_SIDEBAR = {k for k in infos
                      if k.startswith("fw_") or k in ("spf", "gb", "bc")}

# Default picks — one canonical benchmark, one univariate stat model, one structural.
_default_models = [k for k in ["rw", "ao", "ar", "ucsv", "dsge"] if k in infos]

# All model keys that show up in the sidebar (used by the presets to write to
# each checkbox's session-state slot).
_SIDEBAR_KEYS = [k for k in infos if k not in _HIDE_FROM_SIDEBAR]


def _apply_preset(picks: list[str]) -> None:
    """Set every ``pick_<k>`` checkbox state to match ``picks``. Must run BEFORE
    the checkboxes render on this rerun — Streamlit checkboxes ignore the
    ``value=`` argument once their key exists in session_state, so writing
    session_state directly is the only way a preset button can flip them."""
    picks_set = set(picks)
    for k in _SIDEBAR_KEYS:
        st.session_state[f"pick_{k}"] = (k in picks_set)


# Seed defaults on first render.
if "_sidebar_seeded" not in st.session_state:
    _apply_preset(_default_models)
    st.session_state["_sidebar_seeded"] = True

# Presets — one click bulk-selects a coherent set. Because these callbacks
# fire BEFORE the checkboxes render on this rerun, they successfully
# override the widgets' remembered state.
st.sidebar.markdown("### Models to compare")
_p_cols = st.sidebar.columns(3)
if _p_cols[0].button("Basic", help="RW, AO, AR — the classic 3-model benchmark",
                     use_container_width=True):
    _apply_preset([k for k in ["rw", "ao", "ar"] if k in infos])
if _p_cols[1].button("Recommended",
                     help="Balanced mix across the three families",
                     use_container_width=True):
    _apply_preset([k for k in
                   ["rw", "ao", "ucsv", "ar", "nkpc", "tvtnkpc", "dsge", "bb"]
                   if k in infos])
if _p_cols[2].button("Clear", help="Uncheck everything", use_container_width=True):
    _apply_preset([])

# Family-grouped checkboxes. Each family is a collapsible expander so the
# sidebar isn't a wall of 15 items. Read the source-of-truth *from the widget
# states* (which the preset buttons above have already synced), so the
# expander header counts are always accurate.
_current: set[str] = set()
for fam in _FAMILY_ORDER:
    fam_keys = [k for k in _by_family.get(fam, []) if k not in _HIDE_FROM_SIDEBAR]
    if not fam_keys:
        continue
    fam_count = sum(1 for k in fam_keys if st.session_state.get(f"pick_{k}", False))
    with st.sidebar.expander(
        f"{_FAMILY_EMOJI.get(fam, '')} **{fam}** ({fam_count}/{len(fam_keys)})",
        expanded=(fam == "Benchmark"),
    ):
        for k in fam_keys:
            i = infos[k]
            # No ``value=`` — the checkbox owns its state under ``pick_<k>``.
            checked = st.checkbox(i.name, key=f"pick_{k}", help=i.reference)
            if checked:
                _current.add(k)

chosen = [k for k in infos if k in _current]
st.session_state["chosen_models"] = chosen

# --------------------------------------------------------------------------- #
# Per-model customization sliders
# --------------------------------------------------------------------------- #
# For each selected structural model whose parameters we've exposed as
# __init__ kwargs, render a collapsed expander with the right sliders.
# `CUSTOM_SPEC[key]` is a list of (kw_arg, label, min, max, default, step, help).
CUSTOM_SPEC: dict[str, list] = {
    "nkpc": [
        ("anchor_override", "Long-run anchor π^LR (%)",
         0.5, 6.0, None, 0.1,
         "Where the forecast reverts to. Default: Cleveland Fed 10-yr expected "
         "inflation (EXPINF10YR). Move the slider to run a counterfactual — "
         "e.g. 'what if the anchor were 3% instead of 2.5%?'"),
    ],
    "tvtnkpc": [
        ("anchor_override", "Long-run anchor τ (%)",
         0.5, 6.0, None, 0.1,
         "Replaces the EXPINF10YR-derived trend with a constant. Set to "
         "explore anchoring scenarios."),
    ],
    "dsge": [
        ("sigma", "σ — intertemporal elasticity", 0.5, 3.0, 1.0, 0.1,
         "Coefficient on the ex-ante real rate in the IS curve. Higher σ = "
         "spending responds less to real rates. Standard value 1.0 (log utility)."),
        ("kappa", "κ — NKPC slope", 0.005, 0.30, 0.05, 0.005,
         "How strongly the output gap transmits into inflation. Empirical "
         "estimates 0.01-0.10. Higher κ = steeper Phillips curve."),
        ("phi_pi", "φ_π — Taylor rule inflation response", 1.05, 3.0, 1.5, 0.05,
         "Central bank's rate response to a 1pp rise in inflation. Standard "
         "value 1.5 (Taylor). Must exceed 1 for determinacy."),
        ("phi_x", "φ_x — Taylor rule output-gap response", 0.0, 1.0, 0.125, 0.025,
         "Central bank's rate response to a 1pp rise in the output gap. Standard "
         "0.125 (Taylor 1993)."),
    ],
    "nyfed": [
        ("sigma", "σ — intertemporal elasticity", 0.5, 3.0, 1.0, 0.1,
         "Same as in the small NK DSGE."),
        ("kappa", "κ — NKPC slope", 0.005, 0.30, 0.05, 0.005,
         "Higher κ = steeper Phillips curve."),
        ("phi_pi", "φ_π — Taylor rule inflation response", 1.05, 3.0, 1.5, 0.05,
         "Must exceed 1 for determinacy."),
        ("phi_x", "φ_x — Taylor rule output-gap response", 0.0, 1.0, 0.125, 0.025,
         "Standard 0.125."),
    ],
    "sw07": [
        ("crpi", "Taylor rule inflation response (φ_π)", 1.05, 3.0, 2.04, 0.05,
         "SW07 posterior mode = 2.04. Determines how aggressively policy leans "
         "against inflation."),
        ("crr", "Taylor rule interest-rate smoothing (ρ)", 0.0, 0.95, 0.81, 0.05,
         "SW07 posterior mode = 0.81. Higher = more sluggish rate adjustment "
         "(closer to actual Fed behavior)."),
        ("chabb", "Consumption habit (h)", 0.0, 0.95, 0.71, 0.05,
         "SW07 posterior mode = 0.71. Persistence of consumption; higher = "
         "smoother demand response."),
        ("cprobp", "Calvo price stickiness (θ_p)", 0.30, 0.90, 0.66, 0.02,
         "Fraction of firms *unable* to reprice each quarter. SW07 mode = 0.66 "
         "→ average price duration ~ 3 quarters."),
    ],
    "bb": [
        ("n_lags", "Number of lags in wage/price equations", 1, 4, 2, 1,
         "Bernanke-Blanchard (2023) use 4 quarterly lags. Lower = smoother "
         "estimates but less flexible dynamics."),
    ],
}


def _slider_key(model_key: str, kw: str) -> str:
    return f"cust__{model_key}__{kw}"


def _collect_customization(chosen_keys: list[str]) -> dict[str, dict]:
    """Read the current slider values into a {model_key: {kw: value}} dict.
    Only non-default values are recorded so caching stays efficient."""
    out: dict[str, dict] = {}
    for k in chosen_keys:
        if k not in CUSTOM_SPEC:
            continue
        specs = CUSTOM_SPEC[k]
        overrides = {}
        for spec in specs:
            kw, _label, _lo, _hi, default, _step, _help = spec
            state_key = _slider_key(k, kw)
            val = st.session_state.get(state_key)
            if val is None:
                continue
            if default is None:
                # Anchor-override sliders — always send the value (there is no
                # default to compare against; the model itself falls back when
                # anchor_override is None, but the slider is always concrete).
                overrides[kw] = float(val)
            elif val != default:
                overrides[kw] = (int(val) if isinstance(default, int) else float(val))
        if overrides:
            out[k] = overrides
    return out


# Render the customization expander if any selected model has knobs.
# Each model has its OWN "Apply" toggle so you can tweak one model without
# affecting the others — the sliders inside are ignored unless that model's
# toggle is on.
_customizable = [k for k in chosen if k in CUSTOM_SPEC]
_apply_flags: dict[str, bool] = {}
if _customizable:
    st.sidebar.markdown("### ⚙️ Customize model assumptions")
    st.sidebar.caption(
        "Override calibrated parameters for each selected structural model "
        "independently. Each model has its own Apply toggle."
    )
    for k in _customizable:
        info = infos[k]
        apply_key = f"cust_apply_{k}"
        # Show a green dot next to the model name when its customization is on.
        applied_now = st.session_state.get(apply_key, False)
        header = f"🟢 {info.name}" if applied_now else info.name
        with st.sidebar.expander(header):
            apply_it = st.checkbox(
                f"Apply custom values for {info.name}",
                value=applied_now,
                key=apply_key,
                help="When off, this model uses its default (paper) calibration "
                     "regardless of the slider values below.",
            )
            _apply_flags[k] = apply_it
            for kw, label, lo, hi, default, step, help_text in CUSTOM_SPEC[k]:
                skey = _slider_key(k, kw)
                # Initial value: user's previous choice, else the model default,
                # else a sensible midpoint (for anchor overrides where default is None).
                if default is None:
                    init = st.session_state.get(skey, 2.5)
                else:
                    init = st.session_state.get(skey, default)
                if isinstance(default, int):
                    st.slider(label, int(lo), int(hi), int(init), int(step),
                              key=skey, help=help_text)
                else:
                    st.slider(label, float(lo), float(hi), float(init),
                              float(step), key=skey, help=help_text)
            # Reset button — sets every slot for this model back to its default.
            # Uses an on_click callback (fires BEFORE the sliders re-render on
            # the next run), so we're allowed to write the widget-owned keys.
            # Writing them after render inside an `if st.button:` block raises
            # StreamlitAPIException "cannot be modified after the widget was
            # instantiated".
            def _reset_model(model_key=k):
                for kw, _l, _lo, _hi, default, _s, _h in CUSTOM_SPEC[model_key]:
                    st.session_state[_slider_key(model_key, kw)] = (
                        default if default is not None else 2.5
                    )
            st.button(f"Reset {info.name} to defaults", key=f"reset_{k}",
                      on_click=_reset_model)

# Collect the final customization dict — only include models whose Apply
# toggle is on.
_full_custom = _collect_customization(chosen)
_custom_hp = {k: v for k, v in _full_custom.items() if _apply_flags.get(k, False)}


def model_hp_json(key: str) -> str:
    """JSON of hyperparameters for `key`; empty string = defaults."""
    hp = _custom_hp.get(key)
    if not hp:
        return ""
    return json.dumps(hp, sort_keys=True)


horizon = st.sidebar.slider("Forecast horizon (periods ahead)", 1, 24,
                            12 if freq == "M" else 4)
st.sidebar.caption(
    "Slower models (UCSV-SV, DSGE, SW07, NY Fed, SW-DFM, TVP-VAR) run an estimation "
    "step and take a few seconds."
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Inflation Forecasting Dashboard")
st.markdown(
    "Compare canonical **statistical** and **structural** inflation-forecasting models "
    "on a common footing, with pseudo-out-of-sample evaluation against the random-walk "
    "benchmark. Pick models and a horizon in the sidebar."
)

y = data.series(infl_key)
X = data.activity if not data.activity.empty else None
step = pd.DateOffset(months=1 if freq == "M" else 3)

tab_overview, tab_eval, tab_fw, tab_models = st.tabs(
    ["📊 Data & Forecasts", "🎯 Evaluation",
     "🥇 Faust–Wright", "📚 Model Library"]
)

# --------------------------------------------------------------------------- #
# Tab 1 — data + current forecasts
# --------------------------------------------------------------------------- #
with tab_overview:
    # --- Metric tiles (echo the ISMI webapp's card row) ---
    lookback_12 = y.iloc[-12:] if len(y) > 12 else y
    yoy_change = float(y.iloc[-1] - y.iloc[-12]) if len(y) > 12 else float("nan")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Latest {labels.get(infl_key, infl_key)}",
              f"{y.iloc[-1]:.2f}%",
              delta=(f"{y.iloc[-1] - y.iloc[-2]:+.2f} pp" if len(y) > 1 else None),
              help="Annualized inflation rate for the most recent period.")
    m2.metric("12-mo change", f"{yoy_change:+.2f} pp",
              help="Change in the annualized rate over the past 12 periods.")
    m3.metric("Sample", f"{y.index[0]:%Y} → {y.index[-1]:%Y-%m}")
    m4.metric("Observations", f"{len(y):,}")

    # --- Forecast chart ---
    last_date, last_val = y.index[-1], float(y.iloc[-1])
    fc_dates = [last_date + step * h for h in range(1, horizon + 1)]

    # Range-preset UI + a special "Zoom" preset that focuses only on the
    # forecast band with a tight y-axis, so many-model comparisons are readable.
    # Uses st.pills which auto-wraps and never truncates labels — solves the
    # 'Zoom' cut-off-at-the-m problem that narrow columns had.
    RANGE_LABELS = ["🔍 Zoom", "1Y", "2Y", "3Y", "5Y", "10Y", "20Y", "All"]
    RANGE_VALUES = {"🔍 Zoom": "zoom", "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5,
                    "10Y": 10, "20Y": 20, "All": None}

    r_left, r_right = st.columns([3, 1])
    with r_left:
        picked_label = st.pills(
            "Range",
            RANGE_LABELS,
            default="5Y",
            selection_mode="single",
            key="range_pill",
            label_visibility="collapsed",
        ) or "5Y"
    with r_right:
        legend_mode = st.selectbox(
            "Labels",
            ["Inline (right)", "Legend (top)", "Off"],
            index=0 if len(chosen) > 6 else 1,
            key="legend_mode",
            label_visibility="collapsed",
        )

    yrs = RANGE_VALUES[picked_label]
    zoom_mode = (yrs == "zoom")

    # Fit models first — we need the forecast values to compute the zoom range.
    fc_rows = []
    fc_paths = {}   # key -> np.ndarray of forecast values
    with st.spinner("Fitting selected models…"):
        for k in chosen:
            try:
                m = fit_model(freq, infl_key, k, model_hp_json(k))
                path = forecast_path(m, horizon)
                fc_paths[k] = path
                fc_rows.append((infos[k].name, infos[k].family, path[0], path[-1]))
            except Exception as e:
                st.warning(f"{infos[k].name} failed: {e}")

    # Determine the x-range and y-range for the view.
    if zoom_mode:
        # Focus on the forecast band + a short lead-in of history.
        lead_in_months = 6 if freq == "M" else 3      # ~2 quarters lead-in
        view_start = last_date - pd.DateOffset(months=lead_in_months)
        view_end = fc_dates[-1] + pd.DateOffset(months=1 if freq == "M" else 3)
        # Auto-fit y to the *forecast values only* — the realized line is still
        # drawn but is allowed to clip off the top of the chart if today's
        # print is much higher than where the models expect inflation to head.
        # This keeps the vertical scale tight enough to see differences
        # between model paths.
        y_vals = [last_val]
        for path in fc_paths.values():
            y_vals.extend(path.tolist())
        if y_vals:
            y_lo, y_hi = min(y_vals), max(y_vals)
            span = max(0.5, y_hi - y_lo)
            y_range = [y_lo - 0.20 * span, y_hi + 0.20 * span]
        else:
            y_range = None
    elif yrs is None:
        view_start = y.index[0]
        view_end = fc_dates[-1] + pd.DateOffset(months=1 if freq == "M" else 3)
        y_range = None
    else:
        view_start = max(y.index[0], last_date - pd.DateOffset(years=yrs))
        view_end = fc_dates[-1] + pd.DateOffset(months=1 if freq == "M" else 3)
        y_range = None

    fig = go.Figure()

    # Realized inflation. In zoom mode we still draw the line but only over the
    # lead-in window — the user asked to see the actual realized series before
    # the forecast band, not just a marker.
    if zoom_mode:
        realized_view = y[y.index >= view_start]
        fig.add_trace(go.Scatter(
            x=realized_view.index, y=realized_view.values, name="Realized",
            line=dict(color=COLORS["realized"], width=2.5),
            mode="lines+markers",
            marker=dict(size=5),
            showlegend=(legend_mode == "Legend (top)"),
            hovertemplate="%{x|%b %Y}<br><b>%{y:.2f}%</b><extra>Realized</extra>",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=y.index, y=y.values, name="Realized",
            line=dict(color=COLORS["realized"], width=2.5),
            showlegend=(legend_mode == "Legend (top)"),
            hovertemplate="%{x|%b %Y}<br><b>%{y:.2f}%</b><extra>Realized</extra>",
        ))

    # Shaded forecast band + vertical "now" marker
    fig.add_vrect(x0=last_date, x1=fc_dates[-1], fillcolor=COLORS["band"],
                  line_width=0, layer="below")
    fig.add_vline(x=last_date,
                  line=dict(color=COLORS["muted"], width=1, dash="dot"))

    # End-of-line label positions in Inline mode. To reduce collisions when
    # forecasts endpoints stack up, we jitter labels vertically in y-order.
    endpoints = sorted(
        ((k, fc_paths[k][-1]) for k in fc_paths),
        key=lambda x: -x[1],  # top to bottom
    )
    for k in fc_paths:
        path = fc_paths[k]
        color = color_for(k, infos[k].family, chosen)
        show_in_legend = (legend_mode == "Legend (top)")
        # Draw the forecast line
        fig.add_trace(go.Scatter(
            x=[last_date] + fc_dates,
            y=[last_val] + list(path),
            name=infos[k].name, mode="lines",
            line=dict(color=color, width=2.2 if len(chosen) <= 4 else 1.8,
                      dash="dot"),
            showlegend=show_in_legend,
            hovertemplate=(f"%{{x|%b %Y}}<br><b>%{{y:.2f}}%</b>"
                           f"<extra>{infos[k].name}</extra>"),
        ))
        # Inline label at the endpoint
        if legend_mode == "Inline (right)":
            fig.add_annotation(
                x=fc_dates[-1], y=path[-1],
                text=f"  {infos[k].name}",
                showarrow=False, xanchor="left", yanchor="middle",
                font=dict(color=color, size=11),
                bgcolor="rgba(0,0,0,0)",
            )

    # "forecast" text over the shaded region (skip in zoom mode; the label
    # would overlap with the end-of-line names).
    if not zoom_mode:
        fig.add_annotation(
            x=fc_dates[len(fc_dates) // 2], y=1, yref="paper",
            text="forecast horizon", showarrow=False,
            font=dict(color=COLORS["muted"], size=11), yshift=8,
        )

    fig.update_layout(**CHART_LAYOUT)

    # In Inline mode we need extra right-margin space so the endpoint labels
    # aren't clipped. Amount depends on the longest model name.
    inline_pad = 0
    if legend_mode == "Inline (right)":
        max_name = max((len(infos[k].name) for k in fc_paths), default=0)
        inline_pad = min(240, 60 + int(max_name * 6.2))

    fig.update_layout(
        height=520 if zoom_mode else 460,
        margin=dict(t=30, b=40, l=10, r=20 + inline_pad),
        xaxis=dict(**CHART_LAYOUT["xaxis"], type="date",
                   range=[view_start, view_end]),
        yaxis=dict(**CHART_LAYOUT["yaxis"],
                   title=dict(text="Annualized inflation",
                              standoff=8,
                              font=dict(color=COLORS["ink"])),
                   range=y_range),
    )
    # In "Off" mode kill the legend entirely; in "Legend (top)" the horizontal
    # legend from CHART_LAYOUT already handles it.
    if legend_mode == "Off":
        fig.update_layout(showlegend=False)
    elif legend_mode == "Inline (right)":
        fig.update_layout(showlegend=False)

    st.markdown(
        f"#### {labels.get(infl_key, infl_key)} — realized and forecast"
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})
    _caption_prefix = ("Zoomed to the forecast band — y-axis is scaled to the "
                       "model forecasts, so the realized line may clip off the "
                       "chart when today's print is far from where the models "
                       "expect inflation to settle. ") if zoom_mode else \
                       f"Solid line: realized {labels.get(infl_key, infl_key)}. "
    st.caption(
        f"{_caption_prefix}Dotted lines: each model's forecast path from now to "
        f"{horizon} periods ahead. **Zoom** focuses on the forecast band with a "
        "tight y-axis — best for comparing many models. **Labels** toggles between "
        "inline end-of-line names, a legend, or off. Colors: grey = benchmark, "
        "blue = statistical, red = structural."
    )

    # --- Forecast summary table ---
    if fc_rows:
        fcdf = (pd.DataFrame(fc_rows,
                             columns=["Model", "Family", "Next period", f"+{horizon}"])
                .set_index("Model"))
        st.dataframe(
            fcdf.style.format({"Next period": "{:.2f}", f"+{horizon}": "{:.2f}"}),
            use_container_width=True,
        )

    # --- Per-model context cards ---
    if chosen:
        st.subheader("Understanding each forecast")
        st.caption(
            "How each selected model works: its data inputs, key assumptions, "
            "compact math, and what its forecast should look like on the chart."
        )
        for k in chosen:
            info = infos[k]
            color = COLORS.get(info.family, COLORS["muted"])
            with st.expander(f"{info.name}  ·  {info.family}"):
                st.markdown(
                    f"<span style='color:{color};font-weight:600'>{info.reference}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
                if info.citation:
                    st.caption(info.citation)

                # Short about paragraph pulls the description into the expander.
                st.markdown(f"**About** — {info.description}")

                if info.intuition:
                    st.markdown(f"**How it forecasts** — {info.intuition}")
                if info.unique:
                    st.markdown(f"**What makes it different** — {info.unique}")

                # AR(p): expose the lag order actually selected by the fitted model.
                if k == "ar":
                    try:
                        m = fit_model(freq, infl_key, "ar", model_hp_json("ar"))
                        p_star = getattr(m, "_selected_p", None)
                        n_lags = getattr(m, "_num_lags", None)
                        if p_star is not None:
                            st.info(
                                f"**Selected lag order:** p = {p_star}"
                                + (f" ({n_lags} lag{'s' if n_lags != 1 else ''} in the model)"
                                   if n_lags else "")
                                + " — chosen by BIC on the training sample."
                            )
                    except Exception:
                        pass

                # Long-run anchor / steady state — shown for every model. Helps
                # answer the "why is my forecast flat?" question by making the
                # implied long-run destination explicit.
                try:
                    m = fit_model(freq, infl_key, k, model_hp_json(k))
                    ss = None
                    if hasattr(m, "steady_state"):
                        try:
                            ss = float(m.steady_state())
                        except Exception:
                            ss = None
                    if ss is None or not np.isfinite(ss):
                        # As a fallback, use forecast at a very long horizon.
                        try:
                            long_h = 60 if freq == "M" else 20
                            ss = float(m.forecast(long_h))
                        except Exception:
                            ss = None
                    anchor = None
                    anchor_source = None
                    if hasattr(m, "_last_anchor"):
                        anchor = float(m._last_anchor)
                        anchor_source = "EXPINF10YR (Cleveland Fed 10y)"
                    elif hasattr(m, "_trend"):
                        anchor = float(m._trend)
                        anchor_source = getattr(m, "_anchor_source",
                                                "estimated trend τ_T")
                    if ss is not None or anchor is not None:
                        parts = []
                        if ss is not None:
                            parts.append(f"long-run forecast → **{ss:.2f}%**")
                        if anchor is not None:
                            parts.append(f"anchor τ_T = **{anchor:.2f}%**"
                                         + (f" ({anchor_source})" if anchor_source else ""))
                        st.info("**Where this model is heading**: " + " · ".join(parts))
                except Exception:
                    pass

                # Read new fields from EXTRAS directly (belt-and-braces in case
                # Streamlit Cloud has cached a pre-extras ModelInfo class).
                ex = model_extras(k)
                assumptions = ex.get("assumptions") or getattr(info, "assumptions", "")
                equations = ex.get("equations") or getattr(info, "equations", "")
                data_sources = (ex.get("data_sources")
                                or getattr(info, "data_sources", None) or [])

                if assumptions:
                    st.markdown(f"**Key assumptions** — {assumptions}")

                if equations:
                    st.markdown("**Model equations**")
                    st.latex(equations)

                cc1, cc2 = st.columns(2)
                if info.strengths:
                    cc1.markdown(f"**✅ Strengths**\n\n{info.strengths}")
                if info.caveats:
                    cc2.markdown(f"**⚠️ Caveats**\n\n{info.caveats}")
                if info.forecast_shape:
                    st.markdown(f"**Shape on the chart** — {info.forecast_shape}")

                # Data sources.
                if data_sources:
                    st.markdown("**Data sources**")
                    st.markdown(
                        "\n".join(f"- [{label}]({url})" for label, url in data_sources)
                    )

    # --- UCSV-SV decomposition, when selected ---
    if "ucsvsv" in chosen:
        st.subheader("UCSV-SV decomposition (Stock–Watson 2007)")
        st.caption(
            "MCMC estimates of trend inflation and the time-varying volatilities of "
            "the permanent (trend) and transitory shocks."
        )
        m = fit_model(freq, infl_key, "ucsvsv")
        idx = y.index
        d1, d2 = st.columns([3, 2])
        with d1:
            ft = go.Figure()
            ft.add_trace(go.Scatter(x=idx, y=y.values, name="Inflation",
                                    line=dict(color=COLORS["muted"], width=1)))
            ft.add_trace(go.Scatter(x=idx, y=m.trend_path_,
                                    name="Trend τ (posterior mean)",
                                    line=dict(color=COLORS["Structural"], width=2.5)))
            ft.update_layout(**CHART_LAYOUT)
            ft.update_layout(height=320, margin=dict(t=30, b=30, l=10, r=20))
            st.markdown("###### Trend inflation")
            st.plotly_chart(ft, use_container_width=True,
                            config={"displayModeBar": False})
        with d2:
            fv = go.Figure()
            fv.add_trace(go.Scatter(x=idx, y=m.sigma_eta_path_,
                                    name="σ trend (permanent)",
                                    line=dict(color=COLORS["Structural"])))
            fv.add_trace(go.Scatter(x=idx, y=m.sigma_eps_path_,
                                    name="σ transitory",
                                    line=dict(color=COLORS["Statistical"])))
            fv.update_layout(**CHART_LAYOUT)
            fv.update_layout(height=320, margin=dict(t=30, b=30, l=10, r=20))
            st.markdown("###### Stochastic volatility")
            st.plotly_chart(fv, use_container_width=True,
                            config={"displayModeBar": False})

# --------------------------------------------------------------------------- #
# Tab 2 — evaluation / backtest
# --------------------------------------------------------------------------- #
with tab_eval:
    st.markdown("### Pseudo-out-of-sample backtest")
    st.markdown(
        f"""
Recursive backtest: at every past date **t**, each model is re-fit using **only
data available up to t**, then asked to forecast inflation at date **t + {horizon}**.
That forecast is compared to the *actual* value observed **{horizon}** periods later.
Repeat across many origins and score the resulting forecast errors.

- **RMSE** (root-mean-squared error) — the standard forecast error metric.
- **MAE** — mean absolute error, less sensitive to outliers.
- **rel_rmse** — model RMSE divided by the random-walk RMSE. **Values < 1 mean the
  model beats the random walk.**
- **n** — number of forecast/realized pairs used to score.
        """
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        min_train = st.slider(
            "Minimum training window (periods)",
            min_value=60, max_value=int(min(600, max(120, len(y) - horizon - 12))),
            value=min(120, len(y) - horizon - 24), step=12,
            help="Earliest origin: this many observations must be available before "
                 "the first forecast is made.",
        )
    with c2:
        step = st.selectbox(
            "Origin step (periods)", [1, 3, 6, 12], index=1,
            help="Space between successive re-fits. Bigger = faster backtest.",
        )
    with c3:
        _cpu = max(1, (os.cpu_count() or 2))
        _worker_options = sorted(set([1, 2, 4, min(8, _cpu)]))
        bt_workers = st.selectbox(
            "Parallel workers", _worker_options,
            index=min(len(_worker_options) - 1, 2),
            help=f"Distribute origins across processes. Detected {_cpu} CPUs.",
        )

    est_n = max(0, (len(y) - min_train - horizon) // step + 1)
    st.caption(
        f"Plan: **{est_n}** origins × **{max(1, len(chosen))}** models = "
        f"~{est_n * max(1, len(chosen))} model-fits, distributed across "
        f"**{bt_workers}** worker{'s' if bt_workers != 1 else ''}. "
        "Slow models (UCSV-SV, DSGE, SW07, NY Fed, SW-DFM, TVP-VAR) add "
        "several seconds per origin — parallelizing them wins the most."
    )
    run = st.button("Run backtest", type="primary")

    if run:
        if not chosen:
            st.error("Pick at least one model in the sidebar.")
        else:
            keys = (chosen if registry.BENCHMARK_KEY in chosen
                    else [registry.BENCHMARK_KEY] + chosen)
            bar = st.progress(0.0, text="Backtesting…")
            res = run_backtest(
                y, X, keys, horizon=horizon, scheme="expanding",
                min_train=min_train, step=step,
                n_workers=bt_workers,
                progress=lambda p: bar.progress(p, text="Backtesting…"),
            )
            bar.empty()

            lb = res.leaderboard().copy()
            lb.index = [infos[k].name for k in lb.index]
            st.subheader("Leaderboard")

            def _rel_rmse_color(v: float) -> str:
                """Green when the model beats the benchmark (rel_rmse < 1), red
                when worse; anchored at 1 = white. Matplotlib-free so the app
                doesn't need matplotlib in requirements.txt."""
                if pd.isna(v):
                    return ""
                # clamp to [0.5, 1.5] for the gradient; center at 1.0
                x = max(0.5, min(1.5, float(v)))
                if x <= 1.0:
                    # green -> white as x goes 0.5 -> 1.0
                    t = (x - 0.5) / 0.5
                    r = int(200 + t * 55); g = int(240); b = int(200 + t * 55)
                else:
                    # white -> red as x goes 1.0 -> 1.5
                    t = (x - 1.0) / 0.5
                    r = int(255); g = int(240 - t * 100); b = int(255 - t * 155)
                return f"background-color: rgba({r},{g},{b},0.55);"

            st.dataframe(
                lb.style.format({"rmse": "{:.3f}", "mae": "{:.3f}",
                                 "rel_rmse": "{:.3f}", "n": "{:.0f}"})
                        .map(_rel_rmse_color, subset=["rel_rmse"]),
                use_container_width=True,
            )

            # Chart: forecast vs realized, both on the TARGET-date axis so points
            # coincide at the same time (previous versions plotted by origin date,
            # so realized appeared h periods "ahead" of the forecasts — that has
            # been fixed by the new .by_target() helper).
            fc_bt, real_bt = res.by_target()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=real_bt.index, y=real_bt.values, name="Realized",
                line=dict(color=COLORS["realized"], width=2.5),
                hovertemplate="%{x|%b %Y}<br><b>%{y:.2f}%</b><extra>Realized</extra>",
            ))
            for k in keys:
                color = color_for(k, infos[k].family, list(keys))
                fig2.add_trace(go.Scatter(
                    x=fc_bt.index, y=fc_bt[k].values, name=infos[k].name,
                    line=dict(color=color, width=1.6),
                    opacity=0.85,
                    hovertemplate=(f"%{{x|%b %Y}}<br><b>%{{y:.2f}}%</b>"
                                   f"<extra>{infos[k].name}</extra>"),
                ))
            fig2.update_layout(**CHART_LAYOUT)
            fig2.update_layout(
                height=460,
                margin=dict(t=30, b=40, l=10, r=20),
            )
            st.markdown(f"#### Out-of-sample forecasts vs. realized (h = {horizon} periods)")
            st.plotly_chart(fig2, use_container_width=True,
                            config={"displayModeBar": False})

            st.caption(
                "Each dot on a model's line is what that model **would have** forecast "
                "for that target month, using only data available "
                f"{horizon} periods earlier. The black line is what actually happened. "
                "The vertical distance between a model's dot and the black line at the "
                "same date is the forecast error scored in the leaderboard above."
            )
    else:
        st.info("Configure models and horizon in the sidebar, then click "
                "**Run backtest**.")

# --------------------------------------------------------------------------- #
# Tab 3 — Faust–Wright horse race
# (imports at top of file, with an ImportError fallback for stale caches)
# --------------------------------------------------------------------------- #
with tab_fw:
    from src.data.surveys import survey_status
    _sstatus = survey_status()

    # ================================================================= #
    # Header — one sentence of context, one row of stat tiles.
    # ================================================================= #
    st.markdown("## 🥇 Faust–Wright horse race")
    st.caption(
        "Reproduces Faust & Wright's *Forecasting Inflation* (2013) Table 1.2: "
        "compare 20 inflation-forecasting methods, scored by RMSPE relative to their "
        "'fixed ρ = 0.46 AR(1) gap' benchmark. **Below 1.00 = beats the benchmark.**"
    )

    fw_all_keys = [k for k in FW_TABLE_KEYS if k in infos]
    missing_fw = [k for k in FW_TABLE_KEYS if k not in infos]
    _bench_ok = FW_BENCHMARK_KEY in fw_all_keys

    # Compact status strip
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Models available",
              f"{len(fw_all_keys)}",
              help="Total FW Table 1.2 rows in this deploy.")
    survey_count = sum(int(x) for x in (_sstatus.spf_present, _sstatus.gb_present,
                                        _sstatus.bc_available))
    t2.metric("Survey benchmarks", f"{survey_count} / 3",
              help="SPF, Greenbook, Blue-Chip surrogate. FW show these dominate models.")
    t3.metric("Benchmark",
              "Fixed ρ AR(1)" if _bench_ok else "⚠️ missing",
              help="rel_RMSPE divisor for every model.")
    t4.metric("Detected CPUs", f"{max(1, (os.cpu_count() or 2))}",
              help="Used for parallel origin distribution.")

    if missing_fw:
        st.warning(
            f"{len(missing_fw)} FW model(s) not yet in the registry: "
            f"`{'`, `'.join(missing_fw)}`. **Reboot** on Streamlit Cloud to reload."
        )

    # ================================================================= #
    # About / help — collapsed by default.
    # ================================================================= #
    with st.expander("ℹ️ About this exercise & data-file status", expanded=False):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(
                """
**What the horse race does.** For each origin date *t* between 1980 and today,
each model is re-fit using only data up to *t*, then forecasts inflation
*h* quarters ahead. That forecast is compared against the actual inflation
observed *h* quarters later. Repeat across many origins, take the RMSPE.

**What Faust & Wright found:**
- **Subjective survey forecasts** (SPF, Greenbook, Blue-Chip) are the frontier.
- **Gap-form models** — where inflation is decomposed into a slow-moving trend
  τ_t plus a stationary "gap" — dominate stationary specifications.
- The **fixed-ρ AR(1) in gap form** is deceptively hard to beat by more than ~10%.
                """
            )
        with c2:
            st.markdown("**Survey data status**")
            st.markdown(_sstatus.summary())
            if not (_sstatus.spf_present and _sstatus.gb_present):
                st.caption(
                    "Missing files? Download from "
                    "[Philly Fed SPF](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters) "
                    "and [Greenbook](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/greenbook-data-sets) "
                    "→ save as `data/surveys/spf_mean_level.xlsx` and "
                    "`data/surveys/greenbook_row_format.xlsx`."
                )

    # ================================================================= #
    # Configuration — presets + collapsible fine-grained controls.
    # ================================================================= #
    st.markdown("### 1. Configure the run")

    # Presets keep 90% of users out of the multiselect entirely.
    PRESETS = {
        "Quick (fast, ~10-20s)": {
            "keys": [k for k in fw_all_keys
                     if k not in ("sw07", "fw_dsgegap", "tvpvar", "fw_ewa", "fw_bma")],
            "horizons": [0, 1, 4, 8] if freq == "Q" else [0, 1, 6, 12],
            "step": 4,
        },
        "Standard (recommended)": {
            "keys": [k for k in fw_all_keys if k not in ("sw07", "fw_dsgegap", "tvpvar")],
            "horizons": [0, 1, 2, 3, 4, 8] if freq == "Q" else [0, 1, 3, 6, 12, 24],
            "step": 2,
        },
        "Full FW replication (slow)": {
            "keys": list(fw_all_keys),
            "horizons": [0, 1, 2, 3, 4, 8] if freq == "Q" else [0, 1, 3, 6, 12, 24],
            "step": 1,
        },
    }

    pcol, mcol = st.columns([1, 2])
    with pcol:
        preset = st.radio(
            "Preset", list(PRESETS.keys()),
            index=1, key="fw_preset",
            help="A curated bundle of horizons, models, and origin step.",
        )
    with mcol:
        fw_infl = st.selectbox(
            "Inflation measure",
            options=list(data.inflation.columns),
            format_func=lambda k: labels.get(k, k),
            key="fw_infl",
        )
        y_fw = data.series(fw_infl)

    _p = PRESETS[preset]
    with st.expander("Fine-tune (models, horizons, training window, workers)",
                     expanded=False):
        # ---- Models grouped by family ----
        st.markdown("**Models to include**")
        by_family = {"Benchmark": [], "Statistical": [], "Structural": []}
        for k in fw_all_keys:
            by_family.setdefault(infos[k].family, []).append(k)
        selected = set(_p["keys"])
        fam_cols = st.columns(len(by_family))
        for col, fam in zip(fam_cols, ["Benchmark", "Statistical", "Structural"]):
            with col:
                st.caption(f"**{fam}** ({len(by_family.get(fam, []))})")
                for k in by_family.get(fam, []):
                    default = k in selected
                    if st.checkbox(infos[k].name, value=default, key=f"fw_pick_{k}"):
                        selected.add(k)
                    else:
                        selected.discard(k)
        fw_selected = [k for k in fw_all_keys if k in selected]

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            fw_horizons = st.multiselect(
                "Horizons (periods)",
                options=list(range(0, 25)),
                default=_p["horizons"], key="fw_horizons",
            )
        with c2:
            fw_min_train = st.slider(
                "Min training window", 40,
                int(min(400, max(80, len(y_fw) - 30))),
                value=min(80, len(y_fw) - 30), step=4, key="fw_min_train",
            )
        with c3:
            fw_step = st.selectbox("Origin step", [1, 2, 4, 6, 12],
                                   index=[1, 2, 4, 6, 12].index(_p["step"]),
                                   key="fw_step")
        with c4:
            _cpu = max(1, (os.cpu_count() or 2))
            _worker_options = sorted(set([1, 2, 4, min(8, _cpu)]))
            fw_workers = st.selectbox(
                "Parallel workers", _worker_options,
                index=min(len(_worker_options) - 1, 2),
                key="fw_workers",
                help=f"Detected {_cpu} CPUs. More workers = faster.",
            )

    n_origins_est = max(0, (len(y_fw) - fw_min_train - max(fw_horizons or [1])) // fw_step + 1)
    n_fits = n_origins_est * len(fw_selected)

    # Run bar
    rc1, rc2 = st.columns([3, 1])
    with rc1:
        st.caption(
            f"**Plan** — {n_origins_est} origins × {len(fw_selected)} models × "
            f"{len(fw_horizons or [])} horizons = **{n_fits:,} model-fits**. "
            f"Parallelizing across {fw_workers} worker{'s' if fw_workers != 1 else ''}."
        )
    with rc2:
        run_fw = st.button("▶️ Run horse race", type="primary",
                           key="run_fw", use_container_width=True)

    st.markdown("### 2. Results")

    if run_fw:
        if not fw_selected or not fw_horizons:
            st.error("Pick at least one model and one horizon.")
        else:
            keys = (fw_selected if FW_BENCHMARK_KEY in fw_selected
                    else [FW_BENCHMARK_KEY] + fw_selected)
            bar = st.progress(0.0, text="Running horse race…")
            X_fw = data.activity if not data.activity.empty else None
            res = run_fw_horserace(
                y_fw, X_fw, keys, sorted(fw_horizons),
                benchmark_key=FW_BENCHMARK_KEY,
                min_train=fw_min_train, step=fw_step,
                n_workers=fw_workers,
                progress=lambda p: bar.progress(min(1.0, p), text="Running horse race…"),
            )
            bar.empty()
            st.session_state["fw_last_result"] = res
            st.session_state["fw_last_keys"] = keys

    res = st.session_state.get("fw_last_result")
    keys = st.session_state.get("fw_last_keys") or []
    if res is None:
        st.info(
            "▶️ Configure a run above and click **Run horse race**. "
            "The **Standard** preset takes ~10–30s on Streamlit Cloud."
        )
    else:
        # -----------------------------------------------------------------
        # Headline callouts — best/worst at short and long horizons
        # -----------------------------------------------------------------
        short_h = res.horizons[1] if len(res.horizons) > 1 else res.horizons[0]
        long_h = res.horizons[-1]
        rel = res.rel_rmspe.drop(index=[FW_BENCHMARK_KEY], errors="ignore")

        best_short_key = rel[short_h].idxmin() if short_h in rel.columns else None
        best_long_key = rel[long_h].idxmin() if long_h in rel.columns else None
        n_beat_bench = int((rel[long_h] < 1.0).sum()) if long_h in rel.columns else 0

        h1, h2, h3 = st.columns(3)
        if best_short_key is not None and pd.notna(rel.at[best_short_key, short_h]):
            h1.metric(
                f"Best at h={short_h}",
                infos[best_short_key].name,
                delta=f"{(rel.at[best_short_key, short_h] - 1) * 100:+.1f}% vs benchmark",
                delta_color="inverse",
            )
        if best_long_key is not None and pd.notna(rel.at[best_long_key, long_h]):
            h2.metric(
                f"Best at h={long_h}",
                infos[best_long_key].name,
                delta=f"{(rel.at[best_long_key, long_h] - 1) * 100:+.1f}% vs benchmark",
                delta_color="inverse",
            )
        h3.metric(
            f"Beat benchmark at h={long_h}",
            f"{n_beat_bench} / {len(rel)}",
            help="Number of models with rel_RMSPE < 1 at the longest horizon.",
        )

        # -----------------------------------------------------------------
        # Sub-tabs for the different views of the same result
        # -----------------------------------------------------------------
        res_tab_heat, res_tab_lines, res_tab_leader, res_tab_abs = st.tabs(
            ["🔥 Heat map", "📈 Curves", "🏆 Leaderboard", "📏 Absolute RMSPE"]
        )

        def _bg(v):
            if pd.isna(v):
                return ""
            x = max(0.5, min(1.5, float(v)))
            if x <= 1.0:
                t = (x - 0.5) / 0.5
                r = int(200 + t * 55); g = 240; b = int(200 + t * 55)
            else:
                t = (x - 1.0) / 0.5
                r = 255; g = int(240 - t * 100); b = int(255 - t * 155)
            return f"background-color: rgba({r},{g},{b},0.55);"

        display = res.rel_rmspe.copy()
        display.index = [infos[k].name for k in display.index]
        display.columns = [f"h={h}" for h in display.columns]

        with res_tab_heat:
            st.markdown(
                f"**Relative RMSPE vs. {infos[FW_BENCHMARK_KEY].name}.** "
                "Green = beats the benchmark, red = loses. **Sorted by long-horizon "
                "performance** (best at the top)."
            )
            if long_h in res.rel_rmspe.columns:
                order = res.rel_rmspe[long_h].sort_values().index
                display = display.reindex([infos[k].name for k in order])
            st.dataframe(
                display.style.format("{:.2f}").map(_bg),
                use_container_width=True,
            )
            st.caption(
                f"Sample: {int(res.n.iloc[0].max())} origin dates at h={res.horizons[0]}, "
                f"{int(res.n.iloc[0].min())} at h={res.horizons[-1]}."
            )

        with res_tab_lines:
            st.markdown(
                "**Relative RMSPE across horizons.** One line per model — lower is "
                "better. Dashed line at 1.0 = the fixed-ρ benchmark. "
                "Faust–Wright's finding: **subjective forecasts (SPF, Greenbook, BC) "
                "flatten below 1.0 at long horizons**, model-based ones drift upward."
            )
            fig_fw = go.Figure()
            for k in keys:
                # Highlight survey benchmarks with thicker line
                is_survey = k in ("spf", "gb", "bc")
                width = 3 if is_survey else 1.6
                fig_fw.add_trace(go.Scatter(
                    x=[f"h={h}" for h in res.horizons],
                    y=res.rel_rmspe.loc[k].values,
                    name=infos[k].name,
                    line=dict(color=color_for(k, infos[k].family, list(keys)),
                              width=width, dash="solid" if is_survey else "solid"),
                    hovertemplate=f"{infos[k].name}<br>%{{x}}: %{{y:.2f}}<extra></extra>",
                ))
            fig_fw.add_hline(y=1.0, line=dict(color=COLORS["muted"], dash="dot", width=1.5),
                             annotation_text="benchmark = 1.0",
                             annotation_position="top right",
                             annotation_font_color=COLORS["muted"])
            fig_fw.update_layout(**CHART_LAYOUT)
            fig_fw.update_layout(
                height=520, margin=dict(t=30, b=40, l=10, r=20),
                yaxis_title="Relative RMSPE",
            )
            st.plotly_chart(fig_fw, use_container_width=True,
                            config={"displayModeBar": False})

        with res_tab_leader:
            st.markdown(
                f"**Leaderboard at each horizon** — sorted best-to-worst by rel_RMSPE."
            )
            lb_cols = st.columns(min(len(res.horizons), 4))
            for i, h in enumerate(res.horizons):
                col = lb_cols[i % len(lb_cols)]
                with col:
                    st.caption(f"**h = {h}**")
                    lb = res.rel_rmspe[h].sort_values().dropna()
                    lb_df = pd.DataFrame({
                        "Model": [infos[k].name for k in lb.index],
                        "rel_RMSPE": lb.values,
                    })
                    st.dataframe(
                        lb_df.style.format({"rel_RMSPE": "{:.2f}"}).map(
                            _bg, subset=["rel_RMSPE"]
                        ),
                        use_container_width=True, hide_index=True,
                    )

        with res_tab_abs:
            st.markdown(
                "**Absolute RMSPE (percentage points).** Undivided version of the "
                "heat map. Useful when comparing across inflation measures — the "
                "denominator changes but the raw errors stay in the same units."
            )
            abs_df = res.rmspe.copy()
            abs_df.index = [infos[k].name for k in abs_df.index]
            abs_df.columns = [f"h={h}" for h in abs_df.columns]
            st.dataframe(abs_df.style.format("{:.3f}"),
                         use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 4 — model library
# --------------------------------------------------------------------------- #
with tab_models:
    st.markdown(
        "Every model available, grouped by family. This is the reference catalog — "
        "select models in the sidebar to chart and score them."
    )
    for fam in ["Benchmark", "Statistical", "Structural"]:
        fam_infos = [i for i in infos.values() if i.family == fam]
        if not fam_infos:
            continue
        st.subheader(fam)
        for i in fam_infos:
            with st.expander(f"{i.name}  ·  {i.reference}"):
                st.write(i.description)
                if i.citation:
                    st.caption(i.citation)
                if i.unique:
                    st.markdown(f"**Distinctive feature** — {i.unique}")
                # Read the three added fields from EXTRAS directly (belt-and-braces
                # in case Streamlit Cloud has cached a pre-extras ModelInfo class).
                ex = model_extras(i.key)
                assumptions = ex.get("assumptions") or getattr(i, "assumptions", "")
                equations = ex.get("equations") or getattr(i, "equations", "")
                data_sources = (ex.get("data_sources")
                                or getattr(i, "data_sources", None) or [])
                if assumptions:
                    st.markdown(f"**Key assumptions** — {assumptions}")
                if equations:
                    st.markdown("**Model equations**")
                    st.latex(equations)
                if i.needs_activity:
                    st.caption("Uses an activity/slack variable (unemployment gap).")
                if data_sources:
                    st.markdown("**Data sources**")
                    st.markdown(
                        "\n".join(f"- [{label}]({url})" for label, url in data_sources)
                    )
