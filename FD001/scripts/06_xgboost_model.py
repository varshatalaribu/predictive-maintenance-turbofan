"""
Step 7: Train an XGBoost model and compare it against the linear regression baseline.

Run with:
    python3 06_xgboost_model.py
(from inside scripts/, with your venv activated)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("../data/train_FD001_features.csv")

# Same split as the baseline (same seed, same logic) -- this is essential
# for a fair comparison between models.
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

X_train, y_train = train_df[feature_cols], train_df["RUL_clipped"]
X_val, y_val = val_df[feature_cols], val_df["RUL_clipped"]

# --- XGBoost hyperparameters, briefly ---
# n_estimators: how many trees to build in sequence (more trees = more
#   capacity to fit the data, but risk overfitting past a point).
# max_depth: how many yes/no questions deep each individual tree can go
#   (deeper = more complex per-tree patterns, also more overfitting risk).
# learning_rate: how much each new tree is allowed to correct the previous
#   trees' mistakes. Smaller values learn more cautiously/slowly but often
#   generalize better; typically paired with a larger n_estimators.
# subsample / colsample_bytree: each tree is trained on a random 80% of
#   rows / 80% of columns rather than all of them -- this injects useful
#   randomness that reduces overfitting, similar in spirit to how a random
#   forest works.
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
val_preds = model.predict(X_val)
# RUL can't physically be negative -- clip just in case (XGBoost rarely
# predicts negative values the way linear regression did, but this is
# a free safety net either way).
val_preds_clipped = np.clip(val_preds, 0, None)

train_mae = mean_absolute_error(y_train, train_preds)
val_mae = mean_absolute_error(y_val, val_preds)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print("=== XGBoost ===")
print(f"Train MAE:  {train_mae:.2f}")
print(f"Val MAE:    {val_mae:.2f}")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Val RMSE:   {val_rmse:.2f}")

print("\n=== Comparison vs Linear Regression baseline ===")
print("Linear Regression -- Val MAE: 13.37, Val RMSE: 16.84")
print(f"XGBoost           -- Val MAE: {val_mae:.2f}, Val RMSE: {val_rmse:.2f}")

# Feature importance: which engineered features did the model actually rely on?
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 most important features:")
print(importances.head(10))

sample_unit = list(val_units)[0]
sample = val_df[val_df["unit"] == sample_unit].copy()
sample["predicted"] = np.clip(model.predict(sample[feature_cols]), 0, None)
print(f"\nSample engine {sample_unit}, last 10 rows (actual vs predicted RUL):")
print(sample[["cycle", "RUL_clipped", "predicted"]].tail(10).round(1))
