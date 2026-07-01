"""End-to-end smoke test on the synthetic (offline) dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import fred
from src.evaluation.backtest import run_backtest
from src.models import registry

print("live FRED:", fred.has_live_data())
data = fred.load_data(freq="M")
print("synthetic:", data.is_synthetic, "| inflation cols:", list(data.inflation.columns),
      "| activity cols:", list(data.activity.columns))

y = data.series(list(data.inflation.columns)[0])
X = data.activity
print(f"series len={len(y)}  last={y.iloc[-1]:.2f}")

print("\n-- single-fit point forecasts (h=12) --")
for k in registry.MODELS:
    try:
        m = registry.make(k)
        m.fit(y, X)
        print(f"  {k:6s} {m.info.name:38s} -> {m.forecast(12):7.3f}")
    except Exception as e:
        print(f"  {k:6s} FAILED: {type(e).__name__}: {e}")

print("\n-- backtest (h=12, expanding, min_train=180) --")
keys = list(registry.MODELS)
res = run_backtest(y, X, keys, horizon=12, scheme="expanding", min_train=180, step=3)
print(res.leaderboard().to_string(float_format=lambda v: f"{v:.3f}"))
print(f"\norigins evaluated: {len(res.forecasts)}")
print("SMOKE_OK")
