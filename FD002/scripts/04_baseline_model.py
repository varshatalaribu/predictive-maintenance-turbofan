import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("../data/train_FD002_features.csv")

# Engine-grouped split, same principle as FD001: never split by row, or
# near-identical adjacent cycles of the same engine leak across train/val.
np.random.seed(42)
all_units = df["unit"].unique()
np.random.shuffle(all_units)
n_val_units = 40  # ~15% of 260 engines
val_units = set(all_units[:n_val_units])
train_units = set(all_units[n_val_units:])

train_df = df[df["unit"].isin(train_units)].copy()
val_df = df[df["unit"].isin(val_units)].copy()

non_feature_cols = ["unit", "cycle", "RUL", "RUL_clipped"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

model = LinearRegression()
model.fit(train_df[feature_cols], train_df["RUL_clipped"])

train_preds = np.clip(model.predict(train_df[feature_cols]), 0, None)
val_preds = np.clip(model.predict(val_df[feature_cols]), 0, None)

train_mae = mean_absolute_error(train_df["RUL_clipped"], train_preds)
val_mae = mean_absolute_error(val_df["RUL_clipped"], val_preds)

print(f"Train engines: {len(train_units)}, Val engines: {len(val_units)}")
print(f"Train MAE: {train_mae:.2f}")
print(f"Val MAE:   {val_mae:.2f}")
