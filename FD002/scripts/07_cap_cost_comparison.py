import pandas as pd
import numpy as np
import xgboost as xgb

df = pd.read_csv("../data/train_FD002_features.csv")

np.random.seed(42)
all_units = df["unit"].unique()
np.random.shuffle(all_units)
n_val_units = 40
val_units = set(all_units[:n_val_units])
train_units = set(all_units[n_val_units:])

train_df = df[df["unit"].isin(train_units)].copy()
val_df = df[df["unit"].isin(val_units)].sort_values(["unit", "cycle"]).copy()

non_feature_cols = ["unit", "cycle", "RUL", "RUL_clipped"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

C_PLAN = 2000
C_FAIL = 10000
WASTE_RATE = 35

candidate_caps = [30, 40, 50, 125]
thresholds = list(range(1, 21, 1)) + list(range(25, 121, 5))

all_reactive_cost = n_val_units * C_FAIL
summary = []

for cap in candidate_caps:
    y_train = train_df["RUL"].clip(upper=cap)
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(train_df[feature_cols], y_train)

    val_df["predicted_RUL"] = np.clip(model.predict(val_df[feature_cols]), 0, None)

    best_cost = None
    best_threshold = None
    best_caught = None
    best_missed = None
    best_lead = None

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
        if best_cost is None or total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold
            best_caught = n_caught
            best_missed = n_missed
            best_lead = np.mean(lead_times) if lead_times else np.nan

    savings_pct = (1 - best_cost / all_reactive_cost) * 100
    summary.append({
        "cap": cap,
        "optimal_threshold": best_threshold,
        "caught": f"{best_caught}/{n_val_units}",
        "avg_lead_time": round(best_lead, 1) if best_lead == best_lead else None,
        "min_total_cost": best_cost,
        "savings_vs_reactive_pct": round(savings_pct, 1),
    })
    print(f"cap={cap:>4}  best threshold={best_threshold:>3}  caught={best_caught}/{n_val_units}  "
          f"avg lead time={best_lead:.1f}  min cost=${best_cost:,.0f}  savings={savings_pct:.1f}%")

print(f"\nFully reactive baseline (no model): ${all_reactive_cost:,.0f}")
print("\n=== Summary ===")
print(pd.DataFrame(summary).to_string(index=False))
