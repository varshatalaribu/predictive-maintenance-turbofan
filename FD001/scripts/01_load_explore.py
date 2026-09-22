"""
Step 2: Load and explore the FD001 dataset.

Run this whole file top to bottom with:
    python3 01_load_explore.py
(from inside the scripts/ folder, with your venv activated)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The raw files have no header row, so we name the columns ourselves.
# Layout: unit number, time in cycles, 3 operational settings, 21 sensors.
col_names = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# --- Load the training data ---
# - sep=r"\s+"  -> columns are separated by (variable amounts of) whitespace,
#   not a single fixed character like a comma. A plain comma-CSV reader would
#   fail on this file.
# - header=None -> there's no header row in the raw file; we supply col_names ourselves.
train = pd.read_csv(
    "../data/train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=col_names,
)

print("=" * 60)
print("SHAPE")
print("=" * 60)
print("Shape:", train.shape)
print()
print(train.head())
print()

# --- Sanity checks before we trust this data at all ---
print("=" * 60)
print("SANITY CHECKS")
print("=" * 60)
print("Number of unique engines (units):", train["unit"].nunique())
print()
print("Cycles per engine (first 5 engines):")
print(train.groupby("unit")["cycle"].max().head())
print()

# Are there any missing values anywhere? Should be 0 for this dataset.
print("Total missing values:", train.isnull().sum().sum())
print()

# --- Summary statistics ---
# Look especially at columns where std (standard deviation) is 0 or near 0 --
# that means the sensor never changes, so it carries zero information for
# predicting failure and is a candidate to drop later.
print("=" * 60)
print("SUMMARY STATISTICS (columns with std, min, max, etc.)")
print("=" * 60)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(train.describe().T)
print()

print("Done. Next: plot a few engines' sensors over their lifetime.")
