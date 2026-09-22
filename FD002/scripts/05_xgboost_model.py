import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

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

model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
model.fit(train_df[feature_cols], train_df["RUL_clipped"])

train_preds = np.clip(model.predict(train_df[feature_cols]), 0, None)
val_preds = np.clip(model.predict(val_df[feature_cols]), 0, None)

train_mae = mean_absolute_error(train_df["RUL_clipped"], train_preds)
val_mae = mean_absolute_error(val_df["RUL_clipped"], val_preds)

print(f"Train engines: {len(train_units)}, Val engines: {len(val_units)}")
print(f"Train MAE: {train_mae:.2f}")
print(f"Val MAE:   {val_mae:.2f}")
print(f"\n=== Compare to linear regression baseline ===")
print("Linear: Train MAE 12.15, Val MAE 12.70")

# Top features by importance -- sanity check that the model is leaning on
# sensible signals rather than something spurious.
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Top 10 features by importance ===")
print(importances.head(10))
