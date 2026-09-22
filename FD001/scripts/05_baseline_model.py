"""
Step 6: Train/validation split + baseline model (Linear Regression).

Run with:
    python3 05_baseline_model.py
(from inside scripts/, with your venv activated)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("../data/train_FD001_features.csv")
print("Loaded shape:", df.shape)

# --- Split by engine, not by row ---
# A fixed random seed makes this split reproducible: every time this script
# runs, the same 20 engines end up in validation. That matters so we can
# fairly compare this baseline against later, more complex models -- if the
# split changed randomly each run, differences in score could just be due to
# an easier/harder validation set, not the model actually being better.
np.random.seed(42)
all_units = df["unit"].unique()
np.random.shuffle(all_units)

n_val_units = 20
val_units = set(all_units[:n_val_units])
train_units = set(all_units[n_val_units:])

print(f"Train engines: {len(train_units)}, Validation engines: {len(val_units)}")
assert train_units.isdisjoint(val_units), "LEAK: train and val units overlap!"

train_df = df[df["unit"].isin(train_units)].copy()
val_df = df[df["unit"].isin(val_units)].copy()
print(f"Train rows: {len(train_df)}, Validation rows: {len(val_df)}")

# --- Define features (X) and target (y) ---
# Exclude identifier columns and both RUL columns -- RUL (uncapped) must not
# be used as an input feature, since it's directly derived from the target
# we're trying to predict (using it would be leaking the answer).
non_feature_cols = ["unit", "cycle", "RUL", "RUL_clipped"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]
print(f"\nUsing {len(feature_cols)} features")

X_train = train_df[feature_cols]
y_train = train_df["RUL_clipped"]
X_val = val_df[feature_cols]
y_val = val_df["RUL_clipped"]

# --- Train baseline: plain linear regression ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate ---
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)

train_mae = mean_absolute_error(y_train, train_preds)
val_mae = mean_absolute_error(y_val, val_preds)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print("\n=== Baseline: Linear Regression ===")
print(f"Train MAE:  {train_mae:.2f} cycles")
print(f"Val MAE:    {val_mae:.2f} cycles")
print(f"Train RMSE: {train_rmse:.2f} cycles")
print(f"Val RMSE:   {val_rmse:.2f} cycles")

# Sanity check: show actual vs predicted RUL for one validation engine's
# final cycles -- the part of its life that matters most.
sample_unit = list(val_units)[0]
sample = val_df[val_df["unit"] == sample_unit].copy()
sample["predicted"] = model.predict(sample[feature_cols])
print(f"\nSample engine {sample_unit}, last 10 rows (actual vs predicted RUL):")
print(sample[["cycle", "RUL_clipped", "predicted"]].tail(10).round(1))
