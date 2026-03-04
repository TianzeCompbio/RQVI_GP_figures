"""
Figure: RQVI coverage of Flashier GPs as a function of the number of seeds.

For each number of seeds (1–10), computes the fraction of Flashier GPs that
are "covered" (best pseudo-bulked Pearson correlation exceeds threshold) by
the union of RQVI GPs from that many seeds. Seeds are selected greedily:
at each step, the seed that maximises coverage is added.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    PROJECT_DIR, FIG_DIR, CLUSTER_COL, PATH_MAIN_H5AD, load_main_obs,
)

# ─── Paths ───────────────────────────────────────────────────────────────────
CORR_RST_DIR = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/function_analysis/corr_rst"
PATH_FLASHIER_CELL = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/cell_factor_matrix.txt"

N_SEEDS = 10
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ─── Step 1: Compute Flashier cluster means ─────────────────────────────────
print("Loading cell obs for cluster labels...")
obs = load_main_obs()
cluster_labels = obs[CLUSTER_COL]

print("Loading Flashier cell loadings (this may take a minute)...")
flashier_cell = pd.read_csv(PATH_FLASHIER_CELL, sep="\t", index_col=0)
print(f"  Flashier cell matrix: {flashier_cell.shape}")

# Intersect cells present in both
common_cells = flashier_cell.index.intersection(cluster_labels.index)
print(f"  Common cells: {len(common_cells)}")
flashier_cell = flashier_cell.loc[common_cells]
clusters_common = cluster_labels.loc[common_cells]

print("Computing Flashier cluster means...")
flashier_cluster_means = flashier_cell.groupby(clusters_common, observed=True).mean()
print(f"  Flashier cluster means: {flashier_cluster_means.shape}")

n_flashier_gps = flashier_cluster_means.shape[1]

# ─── Step 2: Load RQVI cluster means for all 10 seeds ───────────────────────
print("Loading RQVI cluster means for 10 seeds...")
rqvi_cluster_means = []
for seed in range(N_SEEDS):
    path = f"{CORR_RST_DIR}/rqvi_seed{seed}_gp_cell_level.csv"
    df = pd.read_csv(path, index_col="group")
    df.columns = df.columns.astype(str)
    rqvi_cluster_means.append(df)
    print(f"  Seed {seed}: {df.shape}")

# Align cluster indices across all DataFrames
common_clusters = flashier_cluster_means.index
for df in rqvi_cluster_means:
    common_clusters = common_clusters.intersection(df.index)
common_clusters = sorted(common_clusters)
print(f"  Common clusters across all: {len(common_clusters)}")

flashier_mat = flashier_cluster_means.loc[common_clusters].values  # (n_clusters, n_flashier_gps)
rqvi_mats = [df.loc[common_clusters].values for df in rqvi_cluster_means]  # list of (n_clusters, 256)

# ─── Step 3: Compute best-match correlations ────────────────────────────────
print("Computing pairwise correlations (per seed)...")
# best_corr[seed, flashier_gp] = max abs correlation with any RQVI GP
best_corr = np.zeros((N_SEEDS, n_flashier_gps))

for s in range(N_SEEDS):
    # Correlation matrix: (n_flashier_gps, n_rqvi_gps)
    # Each column of flashier_mat and rqvi_mats[s] is a GP profile across clusters
    # We want Pearson corr between each pair of GP profiles
    F = flashier_mat  # (n_clusters, n_flashier_gps)
    R = rqvi_mats[s]  # (n_clusters, n_rqvi_gps)

    # Standardize columns
    F_z = (F - F.mean(axis=0, keepdims=True)) / (F.std(axis=0, keepdims=True) + 1e-12)
    R_z = (R - R.mean(axis=0, keepdims=True)) / (R.std(axis=0, keepdims=True) + 1e-12)

    # Correlation matrix via dot product
    corr_mat = (F_z.T @ R_z) / F_z.shape[0]  # (n_flashier_gps, n_rqvi_gps)
    best_corr[s] = np.max(np.abs(corr_mat), axis=1)
    print(f"  Seed {s}: median best corr = {np.median(best_corr[s]):.3f}")

# ─── Step 4: Coverage via greedy seed selection ──────────────────────────────
print("Computing coverage via greedy seed selection...")
coverage = np.zeros((N_SEEDS, len(THRESHOLDS)))

for ti, thresh in enumerate(THRESHOLDS):
    covered = np.zeros(n_flashier_gps, dtype=bool)
    remaining = set(range(N_SEEDS))
    for n in range(N_SEEDS):
        # Pick the seed that maximizes coverage
        best_seed, best_cov = None, -1
        for s in remaining:
            new_covered = covered | (best_corr[s] >= thresh)
            cov = new_covered.mean()
            if cov > best_cov:
                best_cov = cov
                best_seed = s
        covered = covered | (best_corr[best_seed] >= thresh)
        remaining.remove(best_seed)
        coverage[n, ti] = covered.mean()
    print(f"  thresh={thresh:.1f}: coverage with 1 seed = {coverage[0, ti]:.3f}, "
          f"all seeds = {coverage[-1, ti]:.3f}")

# ─── Step 5: Plot ────────────────────────────────────────────────────────────
print("Plotting...")
fig, ax = plt.subplots(figsize=(6, 4))

cmap = plt.cm.viridis
colors = [cmap(i / (len(THRESHOLDS) - 1)) for i in range(len(THRESHOLDS))]

x = np.arange(1, N_SEEDS + 1)

for ti, thresh in enumerate(THRESHOLDS):
    y = coverage[:, ti]
    color = colors[ti]
    ax.plot(x, y, marker="o", markersize=4, color=color, linewidth=1.5,
            label=f"r = {thresh:.1f}")

ax.set_xlabel("Number of RQVI seeds", fontsize=11)
ax.set_ylabel("Coverage (fraction of Flashier GPs matched)", fontsize=11)
ax.set_title("RQVI Coverage of Flashier GPs vs. Number of Seeds", fontsize=12,
             fontweight="bold")
ax.set_xticks(x)
ax.set_xlim(0.5, N_SEEDS + 0.5)
ax.set_ylim(-0.02, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend outside
ax.legend(fontsize=8, title="Threshold", title_fontsize=9,
          bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

plt.tight_layout()
outpath = FIG_DIR / "rqvi_flashier_coverage.pdf"
plt.savefig(outpath, bbox_inches="tight", dpi=200)
print(f"Saved to {outpath}")
