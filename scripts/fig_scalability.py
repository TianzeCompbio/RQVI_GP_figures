"""
Figure B: RQVI Scalability — training time vs. number of cells.

Reads benchmark CSV, plots mean +/- std with linear reference line.

Usage:
    python figures_version_v2/scripts/fig_scalability.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import PROJECT_DIR, FIG_DIR

# --- Paths ----------------------------------------------------------------
INPUT_CSV = PROJECT_DIR / "data/scalability_benchmark.csv"
OUTPUT_PDF = FIG_DIR / "scalability.pdf"

# --- Load and aggregate ---------------------------------------------------
df = pd.read_csv(INPUT_CSV)
agg = df.groupby("n_cells")["time_minutes"].agg(["mean", "std"]).reset_index()
agg = agg.sort_values("n_cells")

x = agg["n_cells"].values / 1000  # thousands
y_mean = agg["mean"].values
y_std = agg["std"].values

# --- Linear fit for reference line ----------------------------------------
coeffs = np.polyfit(x, y_mean, 1)
x_fit = np.linspace(x.min() * 0.9, x.max() * 1.05, 100)
y_fit = np.polyval(coeffs, x_fit)

# --- Plot -----------------------------------------------------------------
COLOR = "#2171b5"

fig, ax = plt.subplots(figsize=(7, 5))

# Linear reference
ax.plot(x_fit, y_fit, color="grey", ls="--", lw=1.0, alpha=0.6, label="Linear fit")

# Shaded std region
ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=COLOR, alpha=0.2)

# Main line + markers
ax.plot(x, y_mean, color=COLOR, lw=1.2, zorder=3)
ax.plot(
    x, y_mean,
    marker="o", ms=7, mfc="white", mec=COLOR, mew=1.5,
    ls="none", zorder=4,
)

# Value annotations
for xi, yi, si in zip(x, y_mean, y_std):
    label = f"{yi:.1f}"
    ax.annotate(
        label, (xi, yi),
        textcoords="offset points", xytext=(0, 12),
        ha="center", fontsize=8, color=COLOR,
    )

# Axes
ax.set_xlabel("Number of cells (thousands)", fontsize=12)
ax.set_ylabel("Training time (minutes)", fontsize=12)
ax.set_title("RQVI Training Scalability", fontsize=13, fontweight="bold")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=10)

# Ensure y starts at 0
ax.set_ylim(bottom=0)

ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
fig.savefig(OUTPUT_PDF, dpi=200, bbox_inches="tight")
print(f"Saved to {OUTPUT_PDF}")
plt.close()
