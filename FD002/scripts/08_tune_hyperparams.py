import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer

df = pd.read_csv("../data/train_FD002_features.csv")

non_feature_cols = ["unit", "cycle", "RUL", "RUL_clipped"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

X = df[feature_cols]
y = df["RUL_clipped"]
groups = df["unit"]

DECISION_ZONE = 30

def decision_zone_score(y_true, y_pred):
    mask = y_true <= DECISION_ZONE
    if mask.sum() == 0:
        return -mean_absolute_error(y_true, y_pred)
    return -mean_absolute_error(y_true[mask], np.clip(y_pred[mask], 0, None))

scorer = make_scorer(decision_zone_score, greater_is_better=True)

param_dist = {
    "n_estimators": [100, 150, 200, 300, 400],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
}

base_model = xgb.XGBRegressor(random_state=42, n_jobs=1)
cv = GroupKFold(n_splits=5)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=15,
    scoring=scorer,
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
search.fit(X, y, groups=groups)

print("\nBest params:", search.best_params_)
print(f"Best CV decision-zone MAE: {-search.best_score_:.2f}")

orig_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
cv2 = GroupKFold(n_splits=5)
orig_scores = []
for train_idx, val_idx in cv2.split(X, y, groups=groups):
    orig_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = np.clip(orig_model.predict(X.iloc[val_idx]), 0, None)
    y_true_fold = y.iloc[val_idx]
    mask = y_true_fold <= DECISION_ZONE
    orig_scores.append(mean_absolute_error(y_true_fold[mask], preds[mask]))
print(f"Original hand-picked params, same CV decision-zone MAE: {np.mean(orig_scores):.2f}")

import json
with open("../data/best_hyperparams.json", "w") as f:
    json.dump(search.best_params_, f, indent=2)
print("\nSaved best hyperparameters to ../data/best_hyperparams.json")
