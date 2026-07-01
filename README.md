# Inflation Forecasting Dashboard

An interactive dashboard that reimplements canonical **inflation forecasting models**
from the macro literature (both statistical/reduced-form and structural) behind a
single, easy-to-use interface, with **live data from FRED** and proper
**pseudo-out-of-sample evaluation** against the standard random-walk benchmark.

## Why this exists

Replication packages from AEJ:Macro, QJE, etc. are idiosyncratic, often written in
MATLAB/Dynare/Stata, and are not web-deployable. This project instead reimplements the
*canonical models* against a common `ForecastModel` interface so they can be compared
apples-to-apples, run on current data, and explored online.

## Models

| Family | Model | Reference | Status |
|--------|-------|-----------|--------|
| Benchmark | Random walk (last value) | Atkeson–Ohanian (2001) | ✅ working |
| Benchmark | Atkeson–Ohanian (rolling mean) | Atkeson–Ohanian (2001) | ✅ working |
| Statistical | AR(p) | Stock–Watson (2007) | ✅ working |
| Statistical | Phillips curve | Stock–Watson (1999, 2008) | ✅ working |
| Statistical | UCSV (local-level trend) | Stock–Watson (2007) | ✅ working |
| Statistical | UCSV-SV (stochastic volatility, MCMC) | Stock–Watson (2007) | ✅ working |
| Statistical | BVAR (Minnesota prior, fixed λ) | Litterman (1986); Bańbura et al. (2010) | ✅ working |
| Statistical | BVAR (hierarchical, ML shrinkage) | Giannone–Lenza–Primiceri (2015) | ✅ working |
| Structural | New Keynesian Phillips curve | Galí–Gertler (1999) | ✅ working |
| Structural | Small-scale DSGE (estimated) | Smets–Wouters (2007) tradition | ✅ working |
| Structural | Full Smets–Wouters DSGE (7-shock) | Smets–Wouters (2007) | ✅ working |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: a free FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html)
cp .env.example .env         # then paste your key into FRED_API_KEY

streamlit run app/streamlit_app.py
```

Without a FRED key the app falls back to a bundled **synthetic** inflation series so you
can explore the full pipeline offline.

## Layout

```
src/
  data/fred.py          # FRED fetching, caching, inflation transforms (+ synthetic fallback)
  models/base.py        # ForecastModel interface (fit / forecast / describe)
  models/*.py           # one file per model family
  models/registry.py    # central registry the UI iterates over
  evaluation/backtest.py# recursive/rolling pseudo-OOS backtest + RMSE/MAE vs benchmark
app/streamlit_app.py    # the dashboard
```

## Adding a model

Subclass `ForecastModel` in `src/models/`, implement `fit()` and `forecast(h)`,
and register it in `src/models/registry.py`. It automatically appears in the UI and
the evaluation harness.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (the `.env` file is gitignored and must **not** be committed).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → pick this repository, branch `main`, main file path
   `app/streamlit_app.py`. Under *Advanced settings* choose Python 3.11 or 3.12.
4. In **Secrets**, add your FRED key (TOML format):
   ```toml
   FRED_API_KEY = "your_key_here"
   ```
   The app reads `st.secrets` on the cloud and `.env` locally, so no code changes are
   needed. Without a key it runs on the bundled synthetic series.
5. Deploy. Streamlit installs `requirements.txt` automatically.
