"""
Step 4: Compute the RUL (Remaining Useful Life) label for every row.

Run with:
    python3 03_compute_rul.py
(from inside scripts/, with your venv activated)

Saves the labeled dataset to ../data/train_FD001_with_RUL.csv so later
scripts can just load it directly instead of recomputing this every time.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

col_names = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
train = pd.read_csv("../data/train_FD001.txt", sep=r"\s+", header=None, names=col_names)

# Drop the constant columns identified earlier -- no predictive value.
constant_cols = ["setting_3", "sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
train = train.drop(columns=constant_cols)

# --- Compute RUL ---
# For each engine, find its final (failure) cycle, then subtract the current
# cycle from it. groupby(...).transform("max") broadcasts each engine's max
# cycle back onto every one of that engine's rows -- this is the "grouped"
# operation that respects the fact this is 100 separate time series stacked
# together, not one continuous series.
max_cycle_per_engine = train.groupby("unit")["cycle"].transform("max")
train["RUL"] = max_cycle_per_engine - train["cycle"]

print("=" * 60)
print("RUL FOR ENGINE 1 (first 5 and last 5 rows)")
print("=" * 60)
engine_1 = train[train["unit"] == 1][["unit", "cycle", "RUL"]]
print(engine_1.head())
print("...")
print(engine_1.tail())
print()

print("=" * 60)
print("RUL DISTRIBUTION (raw, uncapped)")
print("=" * 60)
print(train["RUL"].describe())
print()

# --- Clip RUL ---
# Early in an engine's life, sensors look basically identical whether RUL is
# 300 or 280 -- there's no real degradation signal yet, so asking the model
# to distinguish those exact values just adds noise to training. Capping RUL
# at a max value focuses the model's learning on the zone where the signal
# actually exists: the degradation phase close to failure.
RUL_CAP = 125
train["RUL_clipped"] = train["RUL"].clip(upper=RUL_CAP)

print("=" * 60)
print(f"RUL DISTRIBUTION (clipped at {RUL_CAP})")
print("=" * 60)
print(train["RUL_clipped"].describe())
n_affected = (train["RUL"] > RUL_CAP).sum()
print(f"\n{n_affected} of {len(train)} rows ({n_affected/len(train)*100:.1f}%) were capped.")
print()

# Save the labeled dataset for the next steps to reuse.
train.to_csv("../data/train_FD001_with_RUL.csv", index=False)
print("Saved labeled dataset to ../data/train_FD001_with_RUL.csv")

# --- Plot RUL curve for a few engines, to see the shape visually ---
fig, ax = plt.subplots(figsize=(10, 5))
for unit in [1, 2, 3, 4, 5]:
    engine_data = train[train["unit"] == unit]
    ax.plot(engine_data["cycle"], engine_data["RUL_clipped"], label=f"engine {unit}")
ax.set_xlabel("cycle")
ax.set_ylabel("RUL (clipped)")
ax.set_title("Clipped RUL vs. cycle, for 5 example engines")
ax.legend()
plt.tight_layout()
plt.savefig("../plots/rul_curve.png", dpi=120)
print("Saved RUL curve plot to ../plots/rul_curve.png")
