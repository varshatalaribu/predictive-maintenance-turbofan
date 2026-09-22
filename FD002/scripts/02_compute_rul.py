import pandas as pd

train = pd.read_csv("../data/train_FD002_with_clusters.csv")
train = train.sort_values(["unit", "cycle"]).reset_index(drop=True)

max_cycle = train.groupby("unit")["cycle"].transform("max")
train["RUL"] = max_cycle - train["cycle"]
train["RUL_clipped"] = train["RUL"].clip(upper=125)

print("=== RUL summary ===")
print(train[["RUL", "RUL_clipped"]].describe())
print(f"\nRows where RUL was actually capped: {(train['RUL'] > 125).sum()} of {len(train)}")

train.to_csv("../data/train_FD002_with_RUL.csv", index=False)
print("\nSaved to ../data/train_FD002_with_RUL.csv")
