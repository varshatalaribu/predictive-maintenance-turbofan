import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

train = pd.read_csv("../data/train_FD002_features.csv")
test = pd.read_csv("../data/test_FD002_features.csv")
true_rul = pd.read_csv("../data/RUL_FD002.txt", header=None, names=["RUL"])

non_feature_cols = ["unit", "cycle", "RUL", "RUL_clipped"]
feature_cols = [c for c in train.columns if c not in non_feature_cols]

# Train on ALL training engines now (no held-out validation split) -- the
# validation split's job was picking a model/approach; now that we're
# scoring against NASA's real test set, use every training engine we have.
model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
model.fit(train[feature_cols], train["RUL_clipped"])

# Only the LAST cycle of each test engine has a known true RUL (that's what
# RUL_FD002.txt provides -- one number per engine, its remaining life at the
# point the recording was cut off).
last_rows = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
last_rows["true_RUL"] = true_rul["RUL"].values
last_rows["predicted_RUL"] = np.clip(model.predict(last_rows[feature_cols]), 0, None)

mae = mean_absolute_error(last_rows["true_RUL"], last_rows["predicted_RUL"])
rmse = np.sqrt(mean_squared_error(last_rows["true_RUL"], last_rows["predicted_RUL"]))

zone_mask = last_rows["true_RUL"] <= 30
mae_zone = mean_absolute_error(last_rows.loc[zone_mask, "true_RUL"], last_rows.loc[zone_mask, "predicted_RUL"])

print("=== FD002 XGBoost on NASA's held-out test set ===")
print(f"MAE (all {len(last_rows)} engines):  {mae:.2f}")
print(f"RMSE (all {len(last_rows)} engines): {rmse:.2f}")
print(f"MAE (decision zone, true RUL<=30, n={zone_mask.sum()}): {mae_zone:.2f}")

print(f"\n=== Compare to FD001's tuned test-set result ===")
print("FD001 tuned: MAE 10.49 (all engines), decision-zone MAE 4.18")
