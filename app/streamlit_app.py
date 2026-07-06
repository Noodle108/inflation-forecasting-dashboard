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
# Bridge it into the environment before the data layer reads it. (Locally, the key is
# loaded from .env by src/data/fred.py.)
if not os.environ.get("FRED_API_KEY"):
    try:
        if "FRED_API_KEY" in st.secrets:
            os.environ["FRED_API_KEY"] = str(st.secrets["FRED_API_KEY"])
    except Exception:
        pass

from src.data import fred
from src.evaluation.backtest import run_backtest
from src.models import registry

st.set_page_config(page_title="Inflation Forecasting Dashboard", layout="wide", page_icon="📈")

FAMILY_COLORS = {"Benchmark": "#7f8c8d", "Statistical": "#2980b9", "Structural": "#c0392b"}


# --------------------------------------------------------------------------- #
# Cached data + model fits
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading macro data…")
def get_data(freq: str):
    return fred.load_data(freq=freq)


@st.cache_resource(show_spinner=False)
def fit_model(freq: str, infl_key: str, key: str):
    """Fit one model on the selected series and cache it (keyed by inputs)."""
    d = get_data(freq)
    y = d.series(infl_key)
    X = d.activity if not d.activity.empty else None
    m = registry.make(key)
    m.fit(y, X)
    return m


def forecast_path(model, H: int) -> np.ndarray:
    """Vector of h = 1..H forecasts — the trajectory, not just the endpoint."""
    return np.array([model.forecast(h) for h in range(1, H + 1)])


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("📈 Inflation Forecasts")
st.sidebar.caption("Canonical inflation models on live FRED data.")

freq = st.sidebar.radio("Frequency", ["M", "Q"], format_func={"M": "Monthly", "Q": "Quarterly"}.get)
data = get_data(freq)

# Report status based on the data that actually loaded, not just whether a key exists.
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
    "Inflation measure", list(data.inflation.columns), format_func=lambda k: labels.get(k, k)
)

infos = {i.key: i for i in registry.all_infos()}
default_models = [k for k in ["rw", "ao", "ar", "ucsv", "dsge"] if k in infos]
chosen = st.sidebar.multiselect(
    "Models to compare",
    options=list(infos.keys()),
    default=default_models,
    format_func=lambda k: f"{infos[k].name}",
)

horizon = st.sidebar.slider("Forecast horizon (periods ahead)", 1, 24, 12 if freq == "M" else 4)
lookback = st.sidebar.slider("History shown (years)", 1, 25, 5)
st.sidebar.caption("Slower models (UCSV-SV, DSGE, NY Fed DSGE, SW-DFM) run an estimation step and take a few seconds.")

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

tab_overview, tab_eval, tab_models = st.tabs(
    ["📊 Data & Forecasts", "🏆 Evaluation", "📚 Model Library"]
)

# --------------------------------------------------------------------------- #
# Tab 1 — data + current forecasts
# --------------------------------------------------------------------------- #
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Latest {labels.get(infl_key, infl_key)}", f"{y.iloc[-1]:.2f}%",
              help="Annualized inflation rate for the most recent period.")
    c2.metric("Sample", f"{y.index[0]:%Y} → {y.index[-1]:%Y-%m}")
    c3.metric("Observations", f"{len(y):,}")

    # ---- forecast chart: recent window + full h=1..H forecast path per model ----
    cutoff = y.index[-1] - pd.DateOffset(years=lookback)
    y_win = y[y.index >= cutoff]
    last_date, last_val = y.index[-1], float(y.iloc[-1])
    fc_dates = [last_date + step * h for h in range(1, horizon + 1)]

    fig = go.Figure()
    # realized inflation over the chosen window
    fig.add_trace(go.Scatter(x=y_win.index, y=y_win.values, name="Realized inflation",
                             line=dict(color="#333", width=2)))
    # shaded forecast region
    fig.add_vrect(x0=last_date, x1=fc_dates[-1], fillcolor="rgba(0,0,0,0.04)",
                  line_width=0, layer="below")
    fig.add_vline(x=last_date, line=dict(color="#999", width=1, dash="dot"))

    fc_rows = []
    with st.spinner("Fitting selected models…"):
        for k in chosen:
            try:
                m = fit_model(freq, infl_key, k)
                path = forecast_path(m, horizon)
                fc_rows.append((infos[k].name, infos[k].family, path[0], path[-1]))
                color = FAMILY_COLORS.get(infos[k].family, "#555")
                # connect from the anchor point so the path reads continuously
                fig.add_trace(go.Scatter(
                    x=[last_date] + fc_dates, y=[last_val] + list(path),
                    name=infos[k].name, mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    hovertemplate=f"{infos[k].name}<br>%{{x|%Y-%m}}: %{{y:.2f}}%<extra></extra>",
                ))
            except Exception as e:
                st.warning(f"{infos[k].name} failed: {e}")

    fig.update_layout(
        height=470, margin=dict(t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="Annualized inflation (%)",
        xaxis_title=None,
    )
    fig.add_annotation(x=fc_dates[len(fc_dates) // 2], y=1.0, yref="paper",
                       text="forecast", showarrow=False, font=dict(color="#999", size=12))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Solid line: realized {labels.get(infl_key, infl_key)}. Dotted lines: each model's "
        f"forecast path from now to {horizon} periods ahead. Flat paths (Random Walk, "
        f"Atkeson–Ohanian, UCSV) carry a level forward; sloped paths (AR, BVAR, DSGE) "
        f"mean-revert as their dynamics play out. Colors: "
        f"grey = benchmark, blue = statistical, red = structural."
    )

    if fc_rows:
        fcdf = pd.DataFrame(fc_rows, columns=["Model", "Family", "Next period", f"+{horizon}"]).set_index("Model")
        st.dataframe(fcdf.style.format({"Next period": "{:.2f}", f"+{horizon}": "{:.2f}"}),
                     use_container_width=True)

    # ---- per-model context cards ----
    if chosen:
        st.subheader("Understanding each forecast")
        st.caption("What each selected model does and how it differs from the others.")
        for k in chosen:
            info = infos[k]
            color = FAMILY_COLORS.get(info.family, "#555")
            with st.expander(f"{info.name}  ·  {info.family}"):
                st.markdown(f"<span style='color:{color};font-weight:600'>{info.reference}</span>",
                            unsafe_allow_html=True)
                if info.citation:
                    st.caption(info.citation)
                if info.intuition:
                    st.markdown(f"**How it forecasts** — {info.intuition}")
                if info.unique:
                    st.markdown(f"**What makes it different** — {info.unique}")
                cc1, cc2 = st.columns(2)
                if info.strengths:
                    cc1.markdown(f"**✅ Strengths**\n\n{info.strengths}")
                if info.caveats:
                    cc2.markdown(f"**⚠️ Caveats**\n\n{info.caveats}")
                if info.forecast_shape:
                    st.markdown(f"**Shape on the chart** — {info.forecast_shape}")

    # ---- UCSV-SV decomposition, when selected ----
    if "ucsvsv" in chosen:
        st.subheader("UCSV-SV decomposition (Stock–Watson 2007)")
        st.caption(
            "MCMC estimates of trend inflation and the time-varying volatilities of the "
            "permanent (trend) and transitory shocks. High trend volatility marks "
            "episodes when underlying inflation itself was moving (1970s, post-2020)."
        )
        m = fit_model(freq, infl_key, "ucsvsv")
        idx = y.index
        d1, d2 = st.columns([3, 2])
        with d1:
            ft = go.Figure()
            ft.add_trace(go.Scatter(x=idx, y=y.values, name="Inflation",
                                    line=dict(color="#bbb", width=1)))
            ft.add_trace(go.Scatter(x=idx, y=m.trend_path_, name="Trend τ (posterior mean)",
                                    line=dict(color="#c0392b", width=2.5)))
            ft.update_layout(height=320, margin=dict(t=30), title="Trend inflation",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(ft, use_container_width=True)
        with d2:
            fv = go.Figure()
            fv.add_trace(go.Scatter(x=idx, y=m.sigma_eta_path_, name="σ trend (permanent)",
                                    line=dict(color="#c0392b")))
            fv.add_trace(go.Scatter(x=idx, y=m.sigma_eps_path_, name="σ transitory",
                                    line=dict(color="#2980b9")))
            fv.update_layout(height=320, margin=dict(t=30), title="Stochastic volatility",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fv, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 2 — evaluation / backtest
# --------------------------------------------------------------------------- #
with tab_eval:
    st.markdown(
        "Recursive pseudo-out-of-sample backtest. **Relative RMSE < 1** means the model "
        "beats the random-walk benchmark at this horizon."
    )
    min_train = st.slider("Minimum training window", 60, 360, 120, step=12)
    st.caption("Tip: UCSV-SV and DSGE re-estimate at every origin, so a full backtest of "
               "them takes noticeably longer.")
    run = st.button("Run backtest", type="primary")

    if run:
        if not chosen:
            st.error("Pick at least one model in the sidebar.")
        else:
            keys = chosen if registry.BENCHMARK_KEY in chosen else [registry.BENCHMARK_KEY] + chosen
            bar = st.progress(0.0, text="Backtesting…")
            res = run_backtest(
                y, X, keys, horizon=horizon, scheme="expanding",
                min_train=min_train, progress=lambda p: bar.progress(p, text="Backtesting…"),
            )
            bar.empty()

            lb = res.leaderboard().copy()
            lb.index = [infos[k].name for k in lb.index]
            st.subheader("Leaderboard")
            st.dataframe(
                lb.style.format({"rmse": "{:.3f}", "mae": "{:.3f}",
                                 "rel_rmse": "{:.3f}", "n": "{:.0f}"})
                        .background_gradient(subset=["rel_rmse"], cmap="RdYlGn_r"),
                use_container_width=True,
            )

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=res.realized.index, y=res.realized.values,
                                      name="Realized", line=dict(color="#000", width=2)))
            for k in keys:
                fig2.add_trace(go.Scatter(x=res.forecasts.index, y=res.forecasts[k].values,
                                          name=infos[k].name, opacity=0.7,
                                          line=dict(color=FAMILY_COLORS.get(infos[k].family))))
            fig2.update_layout(height=420, margin=dict(t=30),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02),
                               title=f"Out-of-sample forecasts vs realized (h={horizon})")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Configure models and horizon in the sidebar, then click **Run backtest**.")

# --------------------------------------------------------------------------- #
# Tab 3 — model library
# --------------------------------------------------------------------------- #
with tab_models:
    st.markdown("Every model available, grouped by family. This is the reference catalog; "
                "select models in the sidebar to chart and score them.")
    for fam in ["Benchmark", "Statistical", "Structural"]:
        fam_infos = [i for i in infos.values() if i.family == fam]
        if not fam_infos:
            continue
        st.subheader(f"{fam}")
        for i in fam_infos:
            with st.expander(f"{i.name}  ·  {i.reference}"):
                st.write(i.description)
                if i.citation:
                    st.caption(i.citation)
                if i.unique:
                    st.markdown(f"**Distinctive feature** — {i.unique}")
                if i.needs_activity:
                    st.caption("Uses an activity/slack variable (unemployment gap).")
