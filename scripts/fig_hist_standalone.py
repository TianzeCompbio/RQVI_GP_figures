"""
Standalone histogram: multi-seed best-match signed Pearson r distribution.

Computes inline with 10 RQVI seeds + signed correlation (−1 to 1), matching the
coverage figure from fig_representative_pairs.py. Marks GP38/F58 and GP45/F35.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import load_main_obs, FIG_DIR, PROJECT_DIR, CLUSTER_COL

PATH_FLASHIER_CELL = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/cell_factor_matrix.txt"
CORR_RST_DIR = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/function_analysis/corr_rst"


# ─── Step 1: Load metadata & Flashier cluster means ─────────────────────────
print("Loading metadata...")
obs = load_main_obs()

print("Loading Flashier cell loadings...")
flashier_cell = pd.read_csv(PATH_FLASHIER_CELL, sep="\t", index_col=0)
common_cells = flashier_cell.index.intersection(obs.index)
print(f"  Common cells: {len(common_cells)}")

print("Computing Flashier cluster means...")
flashier_sub = flashier_cell.loc[common_cells]
flashier_sub_clusters = obs.loc[common_cells, CLUSTER_COL]
flashier_cluster_means = flashier_sub.groupby(flashier_sub_clusters, observed=True).mean()
print(f"  Flashier cluster means: {flashier_cluster_means.shape}")
del flashier_sub, flashier_cell

# ─── Step 2: Multi-seed best-corr distribution ──────────────────────────────
print("Computing multi-seed best-match correlations...")
N_SEEDS = 10
rqvi_cluster_means_list = []
for seed in range(N_SEEDS):
    path = f"{CORR_RST_DIR}/rqvi_seed{seed}_gp_cell_level.csv"
    df = pd.read_csv(path, index_col="group")
    df.columns = df.columns.astype(str)
    rqvi_cluster_means_list.append(df)

common_clusters = flashier_cluster_means.index
for df in rqvi_cluster_means_list:
    common_clusters = common_clusters.intersection(df.index)
common_clusters = sorted(common_clusters)

F_hist = flashier_cluster_means.loc[common_clusters].values
n_flashier_gps = F_hist.shape[1]
best_corr_per_seed = np.zeros((N_SEEDS, n_flashier_gps))

for s in range(N_SEEDS):
    R_hist = rqvi_cluster_means_list[s].loc[common_clusters].values
    F_z = (F_hist - F_hist.mean(axis=0, keepdims=True)) / (F_hist.std(axis=0, keepdims=True) + 1e-12)
    R_z = (R_hist - R_hist.mean(axis=0, keepdims=True)) / (R_hist.std(axis=0, keepdims=True) + 1e-12)
    corr_mat = (F_z.T @ R_z) / F_z.shape[0]
    best_corr_per_seed[s] = np.max(corr_mat, axis=1)

best_corr_all_seeds = np.max(best_corr_per_seed, axis=0)
pct_covered = (best_corr_all_seeds >= 0.5).mean() * 100
print(f"  {pct_covered:.1f}% of Flashier GPs covered at r>=0.5 (10 seeds)")
del rqvi_cluster_means_list

# ─── Step 3: Plot ────────────────────────────────────────────────────────────
print("Plotting histogram...")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Histogram
ax.hist(best_corr_all_seeds, bins=30,
        color="#4C72B0", edgecolor="white", linewidth=0.5, alpha=0.7,
        zorder=2)

ax.set_xlabel("Best-match Pearson r with RQVI", fontsize=9)
ax.set_ylabel("Count (Flashier factors)", fontsize=9)
ax.set_xlim(-1, 1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ─── Save ────────────────────────────────────────────────────────────────────
outpath = FIG_DIR / "hist_standalone.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=200)
print(f"Saved figure to {outpath}")
plt.close(fig)
print("Done!")
