import pandas as pd
import numpy as np

# FD002-specific: sensors that showed ~0 variance WITHIN a single operating
# condition (i.e. they only ever moved because the condition changed, not
# because of wear) -- narrower than FD001's drop list, since setting_3 and
# sensor_10/sensor_16 turned out to carry real signal here.
DEAD_SENSORS = ["sensor_1", "sensor_5", "sensor_18", "sensor_19"]

train = pd.read_csv("../data/train_FD002_with_RUL.csv")
test = pd.read_csv("../data/test_FD002_with_clusters.csv")
test = test.sort_values(["unit", "cycle"]).reset_index(drop=True)

train = train.drop(columns=DEAD_SENSORS)
test = test.drop(columns=DEAD_SENSORS)

sensor_cols = [c for c in train.columns if c.startswith("sensor_")]

# --- Per-condition normalization ---
# Stats computed from TRAIN ONLY, then applied to both train and test --
# this is the same principle as never letting the test set influence any
# transformation, just like RUL clipping or feature scaling in FD001.
cond_mean = train.groupby("op_condition")[sensor_cols].mean()
cond_std = train.groupby("op_condition")[sensor_cols].std().replace(0, 1.0)  # guard div-by-zero

def normalize(df):
    df = df.copy()
    means = cond_mean.loc[df["op_condition"]].reset_index(drop=True)
    stds = cond_std.loc[df["op_condition"]].reset_index(drop=True)
    for col in sensor_cols:
        df[col] = (df[col].reset_index(drop=True) - means[col]) / stds[col]
    return df

train = normalize(train)
test = normalize(test)

# --- Rolling-window features on the NORMALIZED sensors ---
# Same recipe as FD001 (rolling mean/std/min/max, rate of change, deviation
# from the engine's own early-life baseline), just computed on the
# condition-normalized values instead of raw ones, so a rolling mean isn't
# thrown off by the engine simply switching operating condition mid-window.
def add_rolling_features(df):
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
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

train = add_rolling_features(train)
test = add_rolling_features(test)

# --- One-hot encode operating condition ---
# op_condition is a category (6 regimes), not an ordinal number -- one-hot
# encoding avoids implying condition 5 is "more" of something than condition 0.
train = pd.get_dummies(train, columns=["op_condition"], prefix="cond")
test = pd.get_dummies(test, columns=["op_condition"], prefix="cond")
for col in ["cond_0", "cond_1", "cond_2", "cond_3", "cond_4", "cond_5"]:
    if col not in train.columns:
        train[col] = False
    if col not in test.columns:
        test[col] = False

print("=== Final shapes ===")
print(f"train: {train.shape}, test: {test.shape}")
print(f"\nFeature columns (first 15): {[c for c in train.columns if c not in ('unit','cycle','RUL','RUL_clipped')][:15]}")

train.to_csv("../data/train_FD002_features.csv", index=False)
test.to_csv("../data/test_FD002_features.csv", index=False)

cond_mean.to_csv("../data/cond_mean.csv")
cond_std.to_csv("../data/cond_std.csv")
print("Saved train_FD002_features.csv, test_FD002_features.csv, cond_mean.csv, cond_std.csv")
