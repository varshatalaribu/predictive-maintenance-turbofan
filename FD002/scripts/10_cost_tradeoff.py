import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Final model choice, per 09_final_evaluation.py: hand-picked hyperparameters
# (the tuned ones overfit and did worse on the real test set), cap=125
# (per 07_cap_cost_comparison.py: the cap barely affects actual dollar cost).
FINAL_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
)

df = pd.read_csv("../data/train_FD002_features.csv")

np.random.seed(42)
all_units = df["unit"].unique()
np.random.shuffle(all_units)
n_val_units = 40
val_units = set(all_units[:n_val_units])
train_units = set(all_units[n_val_units:])

train_df = df[df["unit"].isin(train_units)].copy()
val_df = df[df["unit"].isin(val_units)].copy()

non_feature_cols = ["unit", "cycle", "RUL", "RUL_clipped"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

model = xgb.XGBRegressor(**FINAL_PARAMS, random_state=42)
model.fit(train_df[feature_cols], train_df["RUL_clipped"])

val_df = val_df.sort_values(["unit", "cycle"]).copy()
val_df["predicted_RUL"] = np.clip(model.predict(val_df[feature_cols]), 0, None)

C_PLAN = 2000
C_FAIL = 10000
WASTE_RATE = 35

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
                lead_time = trigger_row["RUL"]
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

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(results_df["threshold"], results_df["total_cost"], color="#1f6f5c", linewidth=2)
ax.scatter([best_row["threshold"]], [best_row["total_cost"]], color="#c0392b", zorder=5, s=70,
           label=f"optimal threshold = {int(best_row['threshold'])} cycles")
ax.axhline(all_reactive_cost, color="gray", linestyle="--", linewidth=1,
           label=f"fully reactive (no model): ${all_reactive_cost:,.0f}")
ax.set_xlabel("maintenance trigger threshold (predicted RUL, in cycles)")
ax.set_ylabel("total cost across 40 validation engines ($)")
ax.set_title("FD002 cost tradeoff: acting too early (left) wastes life, too late (right)\nrisks unplanned failure")
ax.legend()
plt.tight_layout()
plt.savefig("../plots/cost_tradeoff.png", dpi=120)
print("\nSaved plot to ../plots/cost_tradeoff.png")

results_df.to_csv("../data/cost_tradeoff_results.csv", index=False)
print("Saved full results table to ../data/cost_tradeoff_results.csv")
