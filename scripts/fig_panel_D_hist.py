"""
Panel D (Histograms): Best-match correlation distributions at cluster and cell level.

Standalone version of the histogram row from fig_panel_D_F.py.
Lightweight — reads only 3 small CSVs, no heavy data loading.

Pairs:
  1. RQVI GP 38 vs Flashier F58
  2. RQVI GP 45 vs Flashier F35
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import FIG_DIR, PROJECT_DIR

# ─── Config ───────────────────────────────────────────────────────────────────
PAIRS = [
    {"rqvi_gp": 38, "flashier_factor": 58, "color": "red", "label": "GP38 vs F58"},
    {"rqvi_gp": 45, "flashier_factor": 35, "color": "#1f77b4", "label": "GP45 vs F35"},
]

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading pre-computed correlation data...")
best_corr_cluster_df = pd.read_csv(PROJECT_DIR / "data" / "cross_method_best_corr.csv")
best_corr_cell_df = pd.read_csv(PROJECT_DIR / "data" / "cross_method_best_corr_cell_level.csv")
pair_corr_df = pd.read_csv(PROJECT_DIR / "data" / "panel_D_F_pair_correlations.csv")

# Attach pair correlations from saved CSV
for p in PAIRS:
    row = pair_corr_df[
        (pair_corr_df["rqvi_gp"] == p["rqvi_gp"])
        & (pair_corr_df["flashier_factor"] == p["flashier_factor"])
    ].iloc[0]
    p["corr_cluster"] = row["corr_cluster"]
    p["corr_cell"] = row["corr_cell"]
    print(f"  {p['label']}: cluster r={p['corr_cluster']:.4f}, cell r={p['corr_cell']:.4f}")

# ─── Plot ─────────────────────────────────────────────────────────────────────
print("Plotting histograms...")
fig, (ax_cl, ax_cell) = plt.subplots(1, 2, figsize=(12, 4))

# Cluster-level histogram
ax_cl.hist(best_corr_cluster_df["best_corr"], bins=30,
           color="#4C72B0", edgecolor="white", linewidth=0.5, alpha=0.7)
for p in PAIRS:
    ax_cl.axvline(p["corr_cluster"], color=p["color"], linestyle="--",
                  linewidth=1.5, label=f"{p['label']} (r={p['corr_cluster']:.2f})")
ax_cl.set_xlabel("Best-match Pearson r (cluster-level)", fontsize=9)
ax_cl.set_ylabel("Count (Flashier factors)", fontsize=9)
ax_cl.set_title("Cluster-level similarity (n=200)", fontsize=10,
                fontweight="bold", loc="left")
ax_cl.legend(fontsize=7, loc="upper left")
ax_cl.spines["top"].set_visible(False)
ax_cl.spines["right"].set_visible(False)

# Cell-level histogram
ax_cell.hist(best_corr_cell_df["best_corr"], bins=30,
             color="#4C72B0", edgecolor="white", linewidth=0.5, alpha=0.7)
for p in PAIRS:
    ax_cell.axvline(p["corr_cell"], color=p["color"], linestyle="--",
                    linewidth=1.5, label=f"{p['label']} (r={p['corr_cell']:.2f})")
ax_cell.set_xlabel("Best-match Pearson r (cell-level)", fontsize=9)
ax_cell.set_ylabel("Count (Flashier factors)", fontsize=9)
ax_cell.set_title("Cell-level similarity (n=200)", fontsize=10,
                  fontweight="bold", loc="left")
ax_cell.legend(fontsize=7, loc="upper left")
ax_cell.spines["top"].set_visible(False)
ax_cell.spines["right"].set_visible(False)

# ─── Save ─────────────────────────────────────────────────────────────────────
outpath = FIG_DIR / "panel_D_hist.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=200)
print(f"Saved figure to {outpath}")
plt.close(fig)
print("Done!")
