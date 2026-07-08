"""Inflation Forecasting Dashboard — Streamlit front end.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

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
from src.models import registry
from src.models.model_extras import EXTRAS as _MODEL_EXTRAS


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
def fit_model(freq: str, infl_key: str, key: str):
    d = get_data(freq)
    y = d.series(infl_key)
    X = d.activity if not d.activity.empty else None
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
default_models = [k for k in ["rw", "ao", "ar", "ucsv", "dsge"] if k in infos]
chosen = st.sidebar.multiselect(
    "Models to compare",
    options=list(infos.keys()),
    default=default_models,
    format_func=lambda k: f"{infos[k].name}",
)

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
    ["📊 Data & Forecasts", "🏆 Evaluation",
     "🐎 Faust–Wright", "📚 Model Library"]
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

    # Range-preset UI as Streamlit buttons: keep the chart itself clean, and let
    # Streamlit re-run when a preset is picked. State survives reruns.
    RANGES = [("1Y", 1), ("2Y", 2), ("3Y", 3), ("5Y", 5),
              ("10Y", 10), ("20Y", 20), ("All", None)]
    if "range_years" not in st.session_state:
        st.session_state["range_years"] = 5    # sensible default

    r_cols = st.columns([1] * len(RANGES) + [6])
    for (label, yrs), col in zip(RANGES, r_cols[:-1]):
        active = st.session_state["range_years"] == yrs
        if col.button(label, key=f"rng_{label}",
                      use_container_width=True,
                      type=("primary" if active else "secondary")):
            st.session_state["range_years"] = yrs

    yrs = st.session_state["range_years"]
    if yrs is None:
        view_start = y.index[0]
    else:
        view_start = max(y.index[0], last_date - pd.DateOffset(years=yrs))
    view_end = fc_dates[-1] + pd.DateOffset(months=1 if freq == "M" else 3)

    fig = go.Figure()

    # Realized inflation
    fig.add_trace(go.Scatter(
        x=y.index, y=y.values, name="Realized inflation",
        line=dict(color=COLORS["realized"], width=2.5),
        hovertemplate="%{x|%b %Y}<br><b>%{y:.2f}%</b><extra>Realized</extra>",
    ))

    # Shaded forecast band + vertical "now" marker
    fig.add_vrect(x0=last_date, x1=fc_dates[-1], fillcolor=COLORS["band"],
                  line_width=0, layer="below")
    fig.add_vline(x=last_date,
                  line=dict(color=COLORS["muted"], width=1, dash="dot"))

    fc_rows = []
    with st.spinner("Fitting selected models…"):
        for k in chosen:
            try:
                m = fit_model(freq, infl_key, k)
                path = forecast_path(m, horizon)
                fc_rows.append((infos[k].name, infos[k].family, path[0], path[-1]))
                color = color_for(k, infos[k].family, chosen)
                fig.add_trace(go.Scatter(
                    x=[last_date] + fc_dates,
                    y=[last_val] + list(path),
                    name=infos[k].name, mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    hovertemplate=(f"%{{x|%b %Y}}<br><b>%{{y:.2f}}%</b>"
                                   f"<extra>{infos[k].name}</extra>"),
                ))
            except Exception as e:
                st.warning(f"{infos[k].name} failed: {e}")

    # "forecast" text over the shaded region
    fig.add_annotation(
        x=fc_dates[len(fc_dates) // 2], y=1, yref="paper",
        text="forecast horizon", showarrow=False,
        font=dict(color=COLORS["muted"], size=11), yshift=8,
    )

    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        height=460,
        margin=dict(t=30, b=40, l=10, r=20),
        xaxis=dict(**CHART_LAYOUT["xaxis"], type="date",
                   range=[view_start, view_end]),
        yaxis=dict(**CHART_LAYOUT["yaxis"],
                   title=dict(text="Annualized inflation",
                              standoff=8,
                              font=dict(color=COLORS["ink"]))),
    )

    # Use a native Streamlit heading instead of Plotly's title, so it picks up
    # the same font/color as the rest of the page (light OR dark theme).
    st.markdown(
        f"#### {labels.get(infl_key, infl_key)} — realized and forecast"
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})
    st.caption(
        f"Solid line: realized {labels.get(infl_key, infl_key)}. Dotted lines: each "
        f"model's forecast path from now to {horizon} periods ahead. Use the range "
        f"buttons above the chart to zoom in and out. Colors: grey = benchmark, "
        f"blue = statistical, red = structural."
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
                        m = fit_model(freq, infl_key, "ar")
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

    c1, c2 = st.columns([2, 1])
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

    est_n = max(0, (len(y) - min_train - horizon) // step + 1)
    st.caption(
        f"Plan: **{est_n}** origins × **{max(1, len(chosen))}** models = "
        f"~{est_n * max(1, len(chosen))} model-fits. Slow models (UCSV-SV, DSGE, SW07, "
        f"NY Fed, SW-DFM, TVP-VAR) can each add several seconds per origin."
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
# --------------------------------------------------------------------------- #
from src.evaluation.fw_horserace import run_fw_horserace
from src.models.registry import FW_BENCHMARK_KEY, FW_TABLE_KEYS

with tab_fw:
    st.markdown("### Faust–Wright (2013) horse race")
    st.markdown(
        """
Faust & Wright's chapter *Forecasting Inflation* (Handbook of Economic Forecasting,
vol. 2A, ch. 1) runs a comprehensive horse race of inflation-forecasting methods.
Their headline object is **Table 1.2**: for each model and each horizon
h = 0, 1, 2, 3, 4, 8 quarters, they report RMSPE relative to a stern benchmark —
an AR(1) in gap form with ρ pinned to 0.46, hand-picked from a 1985-vintage
GDP-deflator fit. **rel_RMSPE < 1 means the model beats the benchmark.**

This tab rebuilds their exercise with the models we can implement on FRED-only
data. Their subjective forecasts (Blue-Chip, SPF, Greenbook) — which they find are
the *frontier* of forecast accuracy — are not on FRED and are omitted.
        """
    )

    with st.expander("Models included (Faust–Wright Table 1.2 rows)"):
        rows = []
        for k in FW_TABLE_KEYS:
            i = infos.get(k)
            if i is None:
                continue
            rows.append({"Table 1.2 row": i.reference.split("—")[-1].strip()
                                          if "—" in i.reference else i.reference,
                         "Model": i.name, "Family": i.family})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Omitted rows: Blue-Chip / SPF / Greenbook subjective forecasts (survey "
            "data not on FRED). Faust–Wright find these are the best forecasters — "
            "they are the frontier every model is trying to reach."
        )

    st.markdown("#### Configuration")
    fw_c1, fw_c2, fw_c3 = st.columns([2, 2, 2])
    with fw_c1:
        fw_infl = st.selectbox(
            "Inflation measure",
            options=list(data.inflation.columns),
            format_func=lambda k: labels.get(k, k),
            key="fw_infl",
        )
    with fw_c2:
        default_horizons = [0, 1, 2, 3, 4, 8] if freq == "Q" else [0, 1, 3, 6, 12, 24]
        fw_horizons = st.multiselect(
            "Horizons (in periods)",
            options=list(range(0, 25)),
            default=default_horizons,
            help=("FW use quarterly horizons 0..8. On the monthly setting we default to "
                  "0,1,3,6,12,24 as the direct analogue."),
        )
    with fw_c3:
        y_fw = data.series(fw_infl)
        fw_min_train = st.slider(
            "Minimum training window",
            min_value=40, max_value=int(min(400, max(80, len(y_fw) - 30))),
            value=min(80, len(y_fw) - 30), step=4,
            key="fw_min_train",
        )
        fw_step = st.selectbox("Origin step", [1, 2, 4, 6, 12], index=2, key="fw_step",
                               help="Bigger step = fewer origins = faster run.")

    fw_all_keys = [k for k in FW_TABLE_KEYS if k in infos]
    default_selected = [k for k in fw_all_keys
                        if k not in ("sw07", "fw_dsgegap", "tvpvar", "fw_ewa")]
    fw_selected = st.multiselect(
        "Models to include (uncheck slow ones — SW07, DSGE-GAP, TVP-VAR, EWA — "
        "if you want a quick run)",
        options=fw_all_keys, default=default_selected,
        format_func=lambda k: infos[k].name, key="fw_selected",
    )

    n_origins_est = max(0, (len(y_fw) - fw_min_train - max(fw_horizons or [1])) // fw_step + 1)
    st.caption(
        f"Plan: **{n_origins_est}** origins × **{len(fw_selected)}** models = "
        f"~{n_origins_est * len(fw_selected)} model-fits at up to "
        f"**{len(fw_horizons or [])}** horizons each. FW's exact exercise ran on "
        f"quarterly data with ~108 origins."
    )
    run_fw = st.button("Run Faust–Wright horse race", type="primary", key="run_fw")

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
                progress=lambda p: bar.progress(min(1.0, p), text="Running horse race…"),
            )
            bar.empty()

            st.markdown(
                f"#### RMSPE relative to benchmark ({infos[FW_BENCHMARK_KEY].name})"
            )
            # Rename rows to human model names
            display = res.rel_rmspe.copy()
            display.index = [infos[k].name for k in display.index]
            display.columns = [f"h={h}" for h in display.columns]

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

            st.dataframe(
                display.style.format("{:.2f}").map(_bg),
                use_container_width=True,
            )
            st.caption(
                "Cells < 1.00 (green) = model beats the benchmark at that horizon; "
                "cells > 1.00 (red) = benchmark beats the model. The benchmark row is "
                "flat 1.00 by construction."
            )

            # Absolute RMSPE table for reference
            with st.expander("Absolute RMSPE (percentage points)"):
                abs_df = res.rmspe.copy()
                abs_df.index = [infos[k].name for k in abs_df.index]
                abs_df.columns = [f"h={h}" for h in abs_df.columns]
                st.dataframe(abs_df.style.format("{:.3f}"),
                             use_container_width=True)

            # Chart: relative RMSPE curves across horizons
            fig_fw = go.Figure()
            for k in keys:
                ys = res.rel_rmspe.loc[k].values
                fig_fw.add_trace(go.Scatter(
                    x=[f"h={h}" for h in res.horizons], y=ys,
                    name=infos[k].name,
                    line=dict(color=color_for(k, infos[k].family, list(keys)),
                              width=2),
                    hovertemplate=f"{infos[k].name}<br>%{{x}}: %{{y:.2f}}<extra></extra>",
                ))
            fig_fw.add_hline(y=1.0, line=dict(color=COLORS["muted"], dash="dot",
                                              width=1),
                             annotation_text="benchmark",
                             annotation_position="right",
                             annotation_font_color=COLORS["muted"])
            fig_fw.update_layout(**CHART_LAYOUT)
            fig_fw.update_layout(height=460, margin=dict(t=30, b=40, l=10, r=20),
                                 yaxis_title="Relative RMSPE")
            st.markdown("#### Relative RMSPE across horizons")
            st.plotly_chart(fig_fw, use_container_width=True,
                            config={"displayModeBar": False})

            st.caption(
                f"n valid pairs per model: {int(res.n.iloc[0].max())} at h={res.horizons[0]}, "
                f"{int(res.n.iloc[0].min())} at h={res.horizons[-1]}. "
                "Faust–Wright's key qualitative findings: (i) subjective forecasts (SPF, "
                "Greenbook, Blue-Chip — not shown here) dominate all model-based ones; "
                "(ii) gap-form models substantially outperform stationary models at "
                "medium and long horizons; (iii) the fixed-ρ benchmark is remarkably "
                "hard to beat by more than ~10%."
            )
    else:
        st.info(
            "Choose a configuration above, then click **Run Faust–Wright horse race**. "
            "A quick run with default settings takes ~30–60 seconds."
        )


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
