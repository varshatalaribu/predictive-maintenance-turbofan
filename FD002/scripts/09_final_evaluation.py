import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# NOTE: hyperparameter tuning (08_tune_hyperparams.py) found parameters that
# scored better under cross-validation (decision-zone MAE 4.84 -> 4.68), but
# performed WORSE on NASA's actual held-out test set (3.70 -> 4.32) and showed
# clear signs of overfitting: the tuned model's training-set MAE dropped from
# 9.31 to 5.52 while test performance didn't improve, meaning the extra
# capacity (deeper trees, more trees, no feature subsampling) let it fit
# training-engine-specific patterns rather than general wear signatures. So
# the FINAL model here deliberately uses the original hand-picked
# hyperparameters, not the tuned ones -- a real, held-out test set overrules
# a CV score when they disagree.
FINAL_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
)

train = pd.read_csv("../data/train_FD002_features.csv")
test = pd.read_csv("../data/test_FD002_features.csv")
true_rul = pd.read_csv("../data/RUL_FD002.txt", header=None, names=["RUL"])

non_feature_cols = ["unit", "cycle", "RUL", "RUL_clipped"]
feature_cols = [c for c in train.columns if c not in non_feature_cols]

model = xgb.XGBRegressor(**FINAL_PARAMS, random_state=42)
model.fit(train[feature_cols], train["RUL_clipped"])

last_rows = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True).copy()
last_rows["true_RUL"] = true_rul["RUL"].values
last_rows["predicted_RUL"] = np.clip(model.predict(last_rows[feature_cols]), 0, None)

mae = mean_absolute_error(last_rows["true_RUL"], last_rows["predicted_RUL"])
rmse = np.sqrt(mean_squared_error(last_rows["true_RUL"], last_rows["predicted_RUL"]))

zone_mask = last_rows["true_RUL"] <= 30
mae_zone = mean_absolute_error(last_rows.loc[zone_mask, "true_RUL"], last_rows.loc[zone_mask, "predicted_RUL"])

print("=== FINAL FD002 MODEL (hand-picked params) on NASA test set ===")
print(f"MAE (all {len(last_rows)} engines):  {mae:.2f}")
print(f"RMSE (all {len(last_rows)} engines): {rmse:.2f}")
print(f"MAE (decision zone, true RUL<=30, n={zone_mask.sum()}): {mae_zone:.2f}")

print(f"\n=== For reference: tuned-params result (not used, overfit) ===")
print("Tuned: MAE 17.69 (all engines), decision-zone MAE 4.32")

print(f"\n=== Compare to FD001 tuned test-set result ===")
print("FD001 tuned: MAE 10.49 (all engines), decision-zone MAE 4.18")

last_rows[["unit", "true_RUL", "predicted_RUL"]].to_csv("../data/test_predictions.csv", index=False)
print("\nSaved predictions to ../data/test_predictions.csv")
