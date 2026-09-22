"""
Step 5: Feature engineering -- turn raw noisy sensor readings into features
that actually expose degradation trends.

Run with:
    python3 04_feature_engineering.py
(from inside scripts/, with your venv activated)

Saves the final feature-engineered dataset to ../data/train_FD001_features.csv
"""

import pandas as pd

col_names = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
train = pd.read_csv("../data/train_FD001.txt", sep=r"\s+", header=None, names=col_names)

constant_cols = ["setting_3", "sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
train = train.drop(columns=constant_cols)

max_cycle = train.groupby("unit")["cycle"].transform("max")
train["RUL"] = max_cycle - train["cycle"]
train["RUL_clipped"] = train["RUL"].clip(upper=125)

# IMPORTANT: sort by unit then cycle first, so rolling windows see cycles in
# the correct chronological order within each engine.
train = train.sort_values(["unit", "cycle"]).reset_index(drop=True)

sensor_cols = [c for c in train.columns if c.startswith("sensor_")]
print(f"Building rolling features for {len(sensor_cols)} sensors...")

grouped = train.groupby("unit")

for col in sensor_cols:
    # Rolling mean over a short window (5 cycles): smooths noise, tracks recent trend.
    train[f"{col}_roll_mean_5"] = grouped[col].transform(
        lambda s: s.rolling(window=5, min_periods=1).mean()
    )
    # Rolling std over the same window: is the sensor becoming more erratic?
    train[f"{col}_roll_std_5"] = grouped[col].transform(
        lambda s: s.rolling(window=5, min_periods=1).std()
    )
    # Rolling mean over a longer window (20 cycles): slower, more reliable trend.
    train[f"{col}_roll_mean_20"] = grouped[col].transform(
        lambda s: s.rolling(window=20, min_periods=1).mean()
    )
    # Rolling min/max over the short window: captures recent extremes, not
    # just the average -- a single spike can matter even if it doesn't move
    # the mean much.
    train[f"{col}_roll_min_5"] = grouped[col].transform(
        lambda s: s.rolling(window=5, min_periods=1).min()
    )
    train[f"{col}_roll_max_5"] = grouped[col].transform(
        lambda s: s.rolling(window=5, min_periods=1).max()
    )
    # Rate of change over 5 cycles: current value minus the value 5 cycles
    # ago. A more direct measure of trend *speed* than the rolling mean,
    # which mostly reflects trend *level*.
    train[f"{col}_rate_of_change_5"] = grouped[col].transform(
        lambda s: s.diff(periods=5)
    )
    # How far has this reading moved from THIS engine's own early-life baseline?
    baseline = grouped[col].transform(lambda s: s.iloc[:5].mean())
    train[f"{col}_diff_from_baseline"] = train[col] - baseline

# roll_std_5 is NaN for each engine's very first cycle (std of 1 value is undefined).
# rate_of_change_5 is NaN for each engine's first 5 cycles (no value 5-cycles-ago yet).
# Both mean "not enough history yet" -- fill with 0 as a reasonable stand-in.
std_cols = [c for c in train.columns if c.endswith("_roll_std_5")]
roc_cols = [c for c in train.columns if c.endswith("_rate_of_change_5")]
train[std_cols + roc_cols] = train[std_cols + roc_cols].fillna(0)

print("Final shape:", train.shape)
print("Remaining NaNs:", train.isnull().sum().sum())
print()
print("Example columns for sensor_2:")
example_cols = [c for c in train.columns if c.startswith("sensor_2_") or c == "sensor_2"]
print(train[["unit", "cycle"] + example_cols].head(8))

train.to_csv("../data/train_FD001_features.csv", index=False)
print()
print("Saved to ../data/train_FD001_features.csv")
