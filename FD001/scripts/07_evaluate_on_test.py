"""
Step 8: Evaluate the final model against NASA's true held-out test set.

This trains on ALL 100 training engines (no need to hold out a validation
split anymore -- the real, honest evaluation now is NASA's separate test
set + RUL_FD001.txt answer key), then predicts RUL at the exact cutoff
point of each of the 100 test engines and compares to the true answer.

Run with:
    python3 07_evaluate_on_test.py
(from inside scripts/, with your venv activated)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

col_names = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
constant_cols = ["setting_3", "sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

def build_features(raw_df):
    """Same feature engineering pipeline as scripts 03/04 -- kept as one
    function here so train and test go through identical processing."""
    df = raw_df.drop(columns=constant_cols).sort_values(["unit", "cycle"]).reset_index(drop=True)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    grouped = df.groupby("unit")
    for col in sensor_cols:
        df[f"{col}_roll_mean_5"] = grouped[col].transform(lambda s: s.rolling(5, min_periods=1).mean())
        df[f"{col}_roll_std_5"] = grouped[col].transform(lambda s: s.rolling(5, min_periods=1).std())
        df[f"{col}_roll_mean_20"] = grouped[col].transform(lambda s: s.rolling(20, min_periods=1).mean())
        df[f"{col}_roll_min_5"] = grouped[col].transform(lambda s: s.rolling(5, min_periods=1).min())
        df[f"{col}_roll_max_5"] = grouped[col].transform(lambda s: s.rolling(5, min_periods=1).max())
        df[f"{col}_rate_of_change_5"] = grouped[col].transform(lambda s: s.diff(periods=5))
        baseline = grouped[col].transform(lambda s: s.iloc[:5].mean())
        df[f"{col}_diff_from_baseline"] = df[col] - baseline
    fill_cols = [c for c in df.columns if c.endswith("_roll_std_5") or c.endswith("_rate_of_change_5")]
    df[fill_cols] = df[fill_cols].fillna(0)
    return df

# --- Train the final model on ALL 100 training engines ---
train_raw = pd.read_csv("../data/train_FD001.txt", sep=r"\s+", header=None, names=col_names)
train = build_features(train_raw)
max_cycle = train.groupby("unit")["cycle"].transform("max")
train["RUL"] = max_cycle - train["cycle"]
train["RUL_clipped"] = train["RUL"].clip(upper=125)

non_feature_cols = ["unit", "cycle", "RUL", "RUL_clipped"]
feature_cols = [c for c in train.columns if c not in non_feature_cols]

model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
model.fit(train[feature_cols], train["RUL_clipped"])
print("Trained on full training set:", train.shape)

# --- Build the same features on the test set ---
test_raw = pd.read_csv("../data/test_FD001.txt", sep=r"\s+", header=None, names=col_names)
test = build_features(test_raw)
print("Test set shape (all rows, all cycles):", test.shape)

# --- Keep only each engine's LAST row -- the cutoff point we predict from ---
last_rows = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
print("Rows after keeping only last cycle per engine:", last_rows.shape, "(should be 100)")

# --- True RUL at that cutoff, from NASA's answer key ---
true_rul = pd.read_csv("../data/RUL_FD001.txt", header=None, names=["RUL"])
assert list(last_rows["unit"]) == list(range(1, 101)), "Unit order mismatch!"

last_rows["true_RUL"] = true_rul["RUL"].values
last_rows["predicted_RUL"] = np.clip(model.predict(last_rows[feature_cols]), 0, None)

mae_raw = mean_absolute_error(last_rows["true_RUL"], last_rows["predicted_RUL"])
rmse_raw = np.sqrt(mean_squared_error(last_rows["true_RUL"], last_rows["predicted_RUL"]))
# Also report MAE against the true RUL clipped at the same 125 cap the model
# was trained against -- an apples-to-apples read of "how well did it learn
# the thing we actually asked it to learn."
mae_fair = mean_absolute_error(last_rows["true_RUL"].clip(upper=125), last_rows["predicted_RUL"])
n_above_cap = (last_rows["true_RUL"] > 125).sum()

print("\n=== Final evaluation on NASA's true held-out test set ===")
print(f"MAE (vs raw true RUL):             {mae_raw:.2f} cycles")
print(f"MAE (vs true RUL clipped at 125):  {mae_fair:.2f} cycles")
print(f"RMSE (vs raw true RUL):            {rmse_raw:.2f} cycles")
print(f"Test engines with true RUL > 125:  {n_above_cap} of {len(last_rows)}")

print("\nFirst 15 engines: true vs predicted RUL")
print(last_rows[["unit", "true_RUL", "predicted_RUL"]].head(15).round(1))

last_rows[["unit", "true_RUL", "predicted_RUL"]].to_csv("../data/test_predictions.csv", index=False)
print("\nSaved per-engine predictions to ../data/test_predictions.csv")
