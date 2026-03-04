"""
Figure: Cross-method similarity between RQVI and Flashier.

For each of the 200 Flashier factors, finds the best-matching RQVI GP (seed 0)
using cluster-aggregated (114 clusters) Pearson correlation. Shows:
  Panel D: Histogram of best-match correlations (200 data points)
  Panel E: Side-by-side UMAP comparisons for the 10 most different pairs

Also saves a cell-level correlation table for reference.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from utils import (
    load_cluster_means, load_main_obs,
    FIG_DIR, PROJECT_DIR, PATH_RQVI_H5AD, PATH_UMAP_CSV,
    LEVEL1_ORDER, CLUSTER_COL,
)

PATH_FLASHIER_CELL = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/cell_factor_matrix.txt"

# ─── Step 1: Load RQVI cluster means (114 clusters × 256 GPs) ───────────────
print("Loading RQVI cluster means...")
obs = load_main_obs()
cluster_means_rqvi = load_cluster_means()  # 114 clusters x 256 GPs
print(f"  RQVI cluster means: {cluster_means_rqvi.shape}")

# ─── Step 2: Compute cluster means for Flashier ─────────────────────────────
print("Loading Flashier cell loadings (this may take a minute)...")
flashier_cell = pd.read_csv(PATH_FLASHIER_CELL, sep="\t", index_col=0)
print(f"  Flashier cell matrix: {flashier_cell.shape}")

common_cells = flashier_cell.index.intersection(obs.index)
print(f"  Common cells: {len(common_cells)}")

flashier_sub = flashier_cell.loc[common_cells].copy()
flashier_sub["cluster"] = obs.loc[common_cells, CLUSTER_COL].values
cluster_means_flashier = flashier_sub.drop(columns=["cluster"]).groupby(flashier_sub["cluster"]).mean()
print(f"  Flashier cluster means: {cluster_means_flashier.shape}")

# Keep only clusters present in both DataFrames
common_clusters = cluster_means_rqvi.index.intersection(cluster_means_flashier.index)
print(f"  Common clusters: {len(common_clusters)}")
cluster_means_rqvi = cluster_means_rqvi.loc[common_clusters]
cluster_means_flashier = cluster_means_flashier.loc[common_clusters]

# ─── Step 3: Compute cluster-level correlation matrix (256 × 200) ────────────
print("Computing cluster-level correlation matrix...")
R = cluster_means_rqvi.values   # (n_clusters, 256)
F = cluster_means_flashier.values  # (n_clusters, 200)
R_z = (R - R.mean(axis=0, keepdims=True)) / (R.std(axis=0, keepdims=True) + 1e-12)
F_z = (F - F.mean(axis=0, keepdims=True)) / (F.std(axis=0, keepdims=True) + 1e-12)
corr_matrix = (R_z.T @ F_z) / R_z.shape[0]  # (256, 200)

n_rqvi = cluster_means_rqvi.shape[1]
n_flashier = cluster_means_flashier.shape[1]
print(f"  Correlation matrix: {corr_matrix.shape}")

# ─── Step 4: Best match for each Flashier factor (cluster-level) ─────────────
best_rqvi_idx = np.argmax(corr_matrix, axis=0)  # 200 values
best_corr = np.max(corr_matrix, axis=0)  # 200 values

flashier_names = cluster_means_flashier.columns.tolist()
rqvi_names = cluster_means_rqvi.columns.tolist()

print(f"  Best-match correlation: median={np.median(best_corr):.3f}, "
      f"min={best_corr.min():.3f}, max={best_corr.max():.3f}")

# Save best-match correlation table as CSV
best_corr_df = pd.DataFrame({
    "flashier_factor": flashier_names,
    "best_rqvi_gp": [rqvi_names[best_rqvi_idx[i]] for i in range(len(flashier_names))],
    "best_corr": best_corr,
})
csv_path = PROJECT_DIR / "data" / "cross_method_best_corr.csv"
best_corr_df.to_csv(csv_path, index=False)
print(f"  Saved cluster-level best-match table to {csv_path}")

# Bottom 10 Flashier factors (most different from RQVI)
bottom10_idx = np.argsort(best_corr)[:10]
print("  Bottom 10 Flashier factors (most different):")
for rank, fi in enumerate(bottom10_idx):
    ri = best_rqvi_idx[fi]
    print(f"    {rank+1}. {flashier_names[fi]} -> RQVI GP {rqvi_names[ri]}, corr={best_corr[fi]:.3f}")

# ─── Step 4b: Cell-level correlation (saved as CSV only) ─────────────────────
print("Computing cell-level correlation matrix...")
adata_rqvi = sc.read_h5ad(PATH_RQVI_H5AD)

# Intersect cells present in both RQVI and Flashier
common_cells_both = pd.Index(common_cells).intersection(adata_rqvi.obs_names)
print(f"  Common cells for cell-level corr: {len(common_cells_both)}")

# Get cell loadings for common cells
rqvi_cell_idx = pd.Index(adata_rqvi.obs_names).get_indexer(common_cells_both)
R_cell = adata_rqvi.X[rqvi_cell_idx, :]  # (n_cells, 256)
if hasattr(R_cell, 'toarray'):
    R_cell = R_cell.toarray()
F_cell = flashier_cell.loc[common_cells_both].values  # (n_cells, 200)

# Z-score across cells and compute correlation
R_cell_z = (R_cell - R_cell.mean(axis=0, keepdims=True)) / (R_cell.std(axis=0, keepdims=True) + 1e-12)
F_cell_z = (F_cell - F_cell.mean(axis=0, keepdims=True)) / (F_cell.std(axis=0, keepdims=True) + 1e-12)
corr_cell = (R_cell_z.T @ F_cell_z) / R_cell_z.shape[0]  # (256, 200)

best_rqvi_idx_cell = np.argmax(corr_cell, axis=0)
best_corr_cell = np.max(corr_cell, axis=0)

rqvi_names_cell = [str(i) for i in range(corr_cell.shape[0])]
flashier_names_cell = flashier_cell.columns.tolist()

best_corr_cell_df = pd.DataFrame({
    "flashier_factor": flashier_names_cell,
    "best_rqvi_gp": [rqvi_names_cell[best_rqvi_idx_cell[i]] for i in range(len(flashier_names_cell))],
    "best_corr": best_corr_cell,
})
csv_cell_path = PROJECT_DIR / "data" / "cross_method_best_corr_cell_level.csv"
best_corr_cell_df.to_csv(csv_cell_path, index=False)
print(f"  Saved cell-level best-match table to {csv_cell_path}")
print(f"  Cell-level best-match correlation: median={np.median(best_corr_cell):.3f}, "
      f"min={best_corr_cell.min():.3f}, max={best_corr_cell.max():.3f}")

# ─── Step 5: Prepare UMAP data ──────────────────────────────────────────────
# Identify the columns we need for UMAP (based on cluster-level bottom 10)
bottom10_flashier_cols = [flashier_names[fi] for fi in bottom10_idx]
bottom10_rqvi_gp_idx = [int(rqvi_names[best_rqvi_idx[fi]]) for fi in bottom10_idx]

# Subset Flashier cell loadings before deleting the big matrix
print("Subsetting Flashier loadings for bottom 10...")
flashier_umap_subset = flashier_cell.loc[common_cells, bottom10_flashier_cols].copy()
del flashier_cell, flashier_sub
import gc; gc.collect()

# Load UMAP coordinates
print("Loading UMAP coordinates...")
umap_df = pd.read_csv(PATH_UMAP_CSV, index_col=0)

# Intersect all three
common_all = (obs.index
              .intersection(umap_df.index)
              .intersection(adata_rqvi.obs_names)
              .intersection(flashier_umap_subset.index))
print(f"  Common cells across all sources: {len(common_all)}")

# Downsample (stratified by level1)
np.random.seed(42)
target_n = 100_000
obs_sub = obs.loc[common_all]
sampled_idx = []
for level1 in LEVEL1_ORDER:
    cells = obs_sub[obs_sub["level1"] == level1].index
    n_sample = max(1, int(len(cells) / len(common_all) * target_n))
    n_sample = min(n_sample, len(cells))
    sampled_idx.extend(np.random.choice(cells, n_sample, replace=False))
sampled_idx = np.array(sampled_idx)
print(f"  Downsampled to {len(sampled_idx)} cells")

umap_coords = umap_df.loc[sampled_idx].values

# Get RQVI loadings for bottom 10 GPs
rqvi_idx_lookup = pd.Index(adata_rqvi.obs_names)
cell_idx_in_rqvi = rqvi_idx_lookup.get_indexer(sampled_idx)
rqvi_loadings = {}
for gp_idx in bottom10_rqvi_gp_idx:
    rqvi_loadings[gp_idx] = adata_rqvi.X[cell_idx_in_rqvi, gp_idx]

# Get Flashier loadings for bottom 10 factors
flashier_loadings = {}
for col in bottom10_flashier_cols:
    flashier_loadings[col] = flashier_umap_subset.loc[sampled_idx, col].values

del adata_rqvi, flashier_umap_subset
gc.collect()

# Shuffle for visual overlap
shuffle_order = np.random.permutation(len(sampled_idx))

# ─── Step 6: Plot ────────────────────────────────────────────────────────────
print("Plotting...")

n_bottom = 10
fig = plt.figure(figsize=(n_bottom * 2.2, 8.5))

# GridSpec: row 0 = histogram (height 3), rows 1-2 = UMAP panels (height 2.5 each)
gs = fig.add_gridspec(3, n_bottom, height_ratios=[3, 2.5, 2.5],
                      hspace=0.35, wspace=0.15)

# ─── Panel D: Histogram ─────────────────────────────────────────────────────
ax_hist = fig.add_subplot(gs[0, :])
ax_hist.hist(best_corr, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)

# Median line
median_corr = np.median(best_corr)
ax_hist.axvline(median_corr, color="black", linestyle="--", linewidth=1.2)
ax_hist.text(median_corr + 0.01, ax_hist.get_ylim()[1] * 0.9,
             f"median = {median_corr:.2f}", fontsize=9, va="top")

# Red rug ticks for bottom 10
for fi in bottom10_idx:
    ax_hist.axvline(best_corr[fi], ymin=0, ymax=0.06, color="red", linewidth=1.5)

ax_hist.set_xlabel("Best-match Pearson correlation (cluster-aggregated)", fontsize=10)
ax_hist.set_ylabel("Count (Flashier factors)", fontsize=10)
ax_hist.set_title("D. Best RQVI match for each Flashier factor (n = 200)", fontsize=11,
                  fontweight="bold", loc="left")
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)

# ─── Panel E: UMAP grid ─────────────────────────────────────────────────────
# Sort bottom 10 by correlation (ascending) for display
sorted_bottom = sorted(range(n_bottom), key=lambda i: best_corr[bottom10_idx[i]])

for col_i, rank_i in enumerate(sorted_bottom):
    fi = bottom10_idx[rank_i]
    ri = best_rqvi_idx[fi]
    fl_name = flashier_names[fi]
    rqvi_gp = int(rqvi_names[ri])
    corr_val = best_corr[fi]

    # Row 1: RQVI GP
    ax_rqvi = fig.add_subplot(gs[1, col_i])
    vals_r = rqvi_loadings[rqvi_gp][shuffle_order]
    vmax_r = np.percentile(vals_r[vals_r > 0], 99) if (vals_r > 0).any() else 1.0
    ax_rqvi.scatter(
        umap_coords[shuffle_order, 0], umap_coords[shuffle_order, 1],
        s=0.3, c=vals_r, cmap="viridis", vmin=0, vmax=vmax_r,
        alpha=0.6, rasterized=True,
    )
    ax_rqvi.set_title(f"RQVI GP {rqvi_gp}", fontsize=7, fontweight="bold")
    ax_rqvi.set_xticks([]); ax_rqvi.set_yticks([])
    for spine in ax_rqvi.spines.values():
        spine.set_visible(False)
    if col_i == 0:
        ax_rqvi.set_ylabel("RQVI", fontsize=9, fontweight="bold")

    # Row 2: Flashier factor
    ax_fl = fig.add_subplot(gs[2, col_i])
    vals_f = flashier_loadings[fl_name][shuffle_order]
    vmax_f = np.percentile(vals_f[vals_f > 0], 99) if (vals_f > 0).any() else 1.0
    ax_fl.scatter(
        umap_coords[shuffle_order, 0], umap_coords[shuffle_order, 1],
        s=0.3, c=vals_f, cmap="viridis", vmin=0, vmax=vmax_f,
        alpha=0.6, rasterized=True,
    )
    ax_fl.set_title(f"Flashier {fl_name}", fontsize=7, fontweight="bold")
    ax_fl.set_xlabel(f"r = {corr_val:.2f}", fontsize=7, color="red")
    ax_fl.set_xticks([]); ax_fl.set_yticks([])
    for spine in ax_fl.spines.values():
        spine.set_visible(False)
    if col_i == 0:
        ax_fl.set_ylabel("Flashier", fontsize=9, fontweight="bold")

# ─── Save ────────────────────────────────────────────────────────────────────
outpath = FIG_DIR / "cross_method_similarity_comprehensive_for_reference.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=200)
print(f"Saved to {outpath}")
plt.close(fig)
print("Done!")
