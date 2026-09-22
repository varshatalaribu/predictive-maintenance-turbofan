"""
Step 10: Cost-tradeoff analysis -- translate RUL predictions into an actual
maintenance-scheduling decision, and quantify the cost of acting too early
vs. too late.

Simulation logic: for each validation engine (which has a full run-to-failure
history, unlike the NASA test set), walk through its life cycle by cycle.
The first cycle where the model's predicted RUL drops to or below a chosen
threshold is when a maintenance crew gets dispatched.
  - If that happens before the engine actually fails: a successful catch --
    but if it happened with a LOT of true remaining life still left, that's
    wasted good life (a cost).
  - If predicted RUL never drops that low before the engine's last recorded
    cycle: a miss -- the engine fails unplanned (an expensive cost).

Run with:
    python3 11_cost_tradeoff.py
(from inside scripts/, with your venv activated)
"""

import json
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("../data/train_FD001_features.csv")

np.random.seed(42)
all_units = df["unit"].unique()
np.random.shuffle(all_units)
n_val_units = 20
val_units = set(all_units[:n_val_units])
train_units = set(all_units[n_val_units:])

train_df = df[df["unit"].isin(train_units)].copy()
val_df = df[df["unit"].isin(val_units)].copy()

non_feature_cols = ["unit", "cycle", "RUL", "RUL_clipped"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

with open("../data/best_hyperparams.json") as f:
    best_params = json.load(f)

model = xgb.XGBRegressor(**best_params, random_state=42)
model.fit(train_df[feature_cols], train_df["RUL_clipped"])

val_df = val_df.sort_values(["unit", "cycle"]).copy()
val_df["predicted_RUL"] = np.clip(model.predict(val_df[feature_cols]), 0, None)

# --- Cost assumptions (edit these to match your own numbers) ---
C_PLAN = 2000      # base cost of one scheduled maintenance visit
C_FAIL = 10000     # cost of one unplanned failure (5x a planned visit)
WASTE_RATE = 35    # extra $ cost per cycle of "good life" thrown away by
                   # acting too early -- the part still had this many cycles
                   # of useful life left when we pulled it for maintenance

thresholds = list(range(1, 21, 1)) + list(range(25, 121, 5))
results = []

for threshold in thresholds:
    n_caught = 0
    n_missed = 0
    lead_times = []
    engine_costs = []
    for unit, engine_data in val_df.groupby("unit"):
        engine_data = engine_data.sort_values("cycle")
        last_cycle = engine_data["cycle"].max()
        triggered = engine_data[engine_data["predicted_RUL"] <= threshold]
        if len(triggered) == 0:
            n_missed += 1
            engine_costs.append(C_FAIL)
        else:
            trigger_row = triggered.iloc[0]
            if trigger_row["cycle"] < last_cycle:
                n_caught += 1
                lead_time = trigger_row["RUL"]  # true cycles remaining at trigger time
                lead_times.append(lead_time)
                engine_costs.append(C_PLAN + WASTE_RATE * lead_time)
            else:
                n_missed += 1
                engine_costs.append(C_FAIL)

    total_cost = sum(engine_costs)
    avg_lead_time = np.mean(lead_times) if lead_times else np.nan
    results.append({
        "threshold": threshold,
        "n_caught": n_caught,
        "n_missed": n_missed,
        "avg_lead_time_cycles": round(avg_lead_time, 1) if lead_times else None,
        "total_cost": total_cost,
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_row = results_df.loc[results_df["total_cost"].idxmin()]
all_reactive_cost = n_val_units * C_FAIL

print(f"\nOptimal threshold: {int(best_row['threshold'])} cycles")
print(f"  Caught: {int(best_row['n_caught'])}/{n_val_units}, Missed: {int(best_row['n_missed'])}/{n_val_units}")
print(f"  Avg lead time when caught: {best_row['avg_lead_time_cycles']} cycles before failure")
print(f"  Total cost: ${best_row['total_cost']:,.0f}")
print(f"  Compare to a fully reactive strategy (fix only after failure): ${all_reactive_cost:,.0f}")
print(f"  Savings: ${all_reactive_cost - best_row['total_cost']:,.0f} ({(1 - best_row['total_cost']/all_reactive_cost)*100:.0f}% reduction)")

# --- Plot the cost curve ---
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(results_df["threshold"], results_df["total_cost"], color="#1f6f5c", linewidth=2)
ax.scatter([best_row["threshold"]], [best_row["total_cost"]], color="#c0392b", zorder=5, s=70,
           label=f"optimal threshold = {int(best_row['threshold'])} cycles")
ax.axhline(all_reactive_cost, color="gray", linestyle="--", linewidth=1,
           label=f"fully reactive (no model): ${all_reactive_cost:,.0f}")
ax.set_xlabel("maintenance trigger threshold (predicted RUL, in cycles)")
ax.set_ylabel("total cost across 20 validation engines ($)")
ax.set_title("Cost tradeoff: acting too early (left) wastes life, too late (right)\nrisks unplanned failure")
ax.legend()
plt.tight_layout()
plt.savefig("../plots/cost_tradeoff.png", dpi=120)
print("\nSaved plot to ../plots/cost_tradeoff.png")

results_df.to_csv("../data/cost_tradeoff_results.csv", index=False)
print("Saved full results table to ../data/cost_tradeoff_results.csv")
