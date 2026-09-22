"""
Step 9b: Final evaluation of the tuned model on NASA's true test set,
compared against the original hand-picked-hyperparameter model.

Run with:
    python3 10_final_evaluation.py
(from inside scripts/, with your venv activated, after running
09_tune_hyperparams.py at least once to produce best_hyperparams.json)
"""

import json
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

def evaluate(model, train, test_last_rows, feature_cols, label):
    preds = np.clip(model.predict(test_last_rows[feature_cols]), 0, None)
    mae = mean_absolute_error(test_last_rows["true_RUL"], preds)
    rmse = np.sqrt(mean_squared_error(test_last_rows["true_RUL"], preds))
    zone_mask = test_last_rows["true_RUL"] <= 30
    mae_zone = mean_absolute_error(test_last_rows.loc[zone_mask, "true_RUL"], preds[zone_mask])
    print(f"{label:>10} -- MAE (all): {mae:5.2f}   RMSE (all): {rmse:5.2f}   MAE (decision zone, n={zone_mask.sum()}): {mae_zone:.2f}")
    return mae, mae_zone

train_raw = pd.read_csv("../data/train_FD001.txt", sep=r"\s+", header=None, names=col_names)
train = build_features(train_raw)
max_cycle = train.groupby("unit")["cycle"].transform("max")
train["RUL"] = max_cycle - train["cycle"]
train["RUL_clipped"] = train["RUL"].clip(upper=125)

non_feature_cols = ["unit", "cycle", "RUL", "RUL_clipped"]
feature_cols = [c for c in train.columns if c not in non_feature_cols]

test_raw = pd.read_csv("../data/test_FD001.txt", sep=r"\s+", header=None, names=col_names)
test = build_features(test_raw)
last_rows = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
true_rul = pd.read_csv("../data/RUL_FD001.txt", header=None, names=["RUL"])
last_rows["true_RUL"] = true_rul["RUL"].values

print("=== Comparing original vs. tuned model on NASA's true test set ===\n")

# Original hand-picked hyperparameters
orig_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
orig_model.fit(train[feature_cols], train["RUL_clipped"])
evaluate(orig_model, train, last_rows, feature_cols, "Original")

# Tuned hyperparameters from the cross-validated search
with open("../data/best_hyperparams.json") as f:
    best_params = json.load(f)
tuned_model = xgb.XGBRegressor(**best_params, random_state=42)
tuned_model.fit(train[feature_cols], train["RUL_clipped"])
evaluate(tuned_model, train, last_rows, feature_cols, "Tuned")

print(f"\nTuned params used: {best_params}")
