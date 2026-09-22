import pandas as pd
from sklearn.cluster import KMeans

col_names = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

train = pd.read_csv("../data/train_FD002.txt", sep=r"\s+", header=None, names=col_names)
test = pd.read_csv("../data/test_FD002.txt", sep=r"\s+", header=None, names=col_names)
true_rul = pd.read_csv("../data/RUL_FD002.txt", header=None, names=["RUL"])

print("=== Shape ===")
print(f"train: {train.shape}, test: {test.shape}, true_rul: {true_rul.shape}")
print(f"train engines: {train['unit'].nunique()}, test engines: {test['unit'].nunique()}")

# --- Identify the 6 operating conditions ---
# Unlike FD001 (1 condition), FD002 engines cycle through 6 combinations of
# setting_1/2/3 (altitude, Mach number, throttle resolver angle). We cluster
# on the 3 settings to assign each row a discrete operating-condition label.
km = KMeans(n_clusters=6, n_init=10, random_state=42)
train["op_condition"] = km.fit_predict(train[["setting_1", "setting_2", "setting_3"]])
test["op_condition"] = km.predict(test[["setting_1", "setting_2", "setting_3"]])

print("\n=== Operating condition cluster sizes (train) ===")
print(train["op_condition"].value_counts().sort_index())
print("\n=== Mean settings per cluster ===")
print(train.groupby("op_condition")[["setting_1", "setting_2", "setting_3"]].mean().round(2))

# --- Sensor variance check ---
# In FD001, several sensors were constant (no info) because there was only
# one operating condition. In FD002 the same sensors may now show variance
# ACROSS conditions even if they're still flat WITHIN a single condition --
# check both.
print("\n=== Sensor std, whole dataset (mixes conditions) ===")
print(train.filter(like="sensor_").std().sort_values().round(4))

print("\n=== Sensor std, WITHIN each condition (averaged across conditions) ===")
within_std = train.groupby("op_condition")[[c for c in train.columns if c.startswith("sensor_")]].std().mean()
print(within_std.sort_values().round(4))

train.to_csv("../data/train_FD002_with_clusters.csv", index=False)
test.to_csv("../data/test_FD002_with_clusters.csv", index=False)
print("\nSaved train/test with op_condition labels to ../data/")
