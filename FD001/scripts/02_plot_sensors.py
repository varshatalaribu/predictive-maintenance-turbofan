"""
Step 3: Visualize sensor degradation over an engine's lifetime.

Run with:
    python3 02_plot_sensors.py
(from inside scripts/, with your venv activated)

Saves a PNG to ../plots/sensor_trends.png -- open that file to view it.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # write straight to a file instead of popping up a window
import matplotlib.pyplot as plt

col_names = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
train = pd.read_csv("../data/train_FD001.txt", sep=r"\s+", header=None, names=col_names)

# Drop the constant columns we identified in step 2 -- zero variance means
# zero predictive power, so there's no reason to keep dragging them along.
constant_cols = ["setting_3", "sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
train = train.drop(columns=constant_cols)
print(f"Dropped {len(constant_cols)} constant columns: {constant_cols}")

# A handful of sensors that show visible degradation trends in FD001.
sensors_to_plot = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]

# Plot a few different engines on the same axes so you can compare how their
# degradation trajectories differ in speed/shape, even though they're the
# same engine model under the same operating condition.
engines_to_plot = [1, 2, 3]

fig, axes = plt.subplots(len(sensors_to_plot), 1, figsize=(10, 14))
for ax, sensor in zip(axes, sensors_to_plot):
    for unit in engines_to_plot:
        engine_data = train[train["unit"] == unit]
        ax.plot(engine_data["cycle"], engine_data[sensor], label=f"engine {unit}", linewidth=1)
    ax.set_title(sensor)
    ax.set_xlabel("cycle")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("../plots/sensor_trends.png", dpi=120)
print("Saved plot to ../plots/sensor_trends.png -- open it to view.")
