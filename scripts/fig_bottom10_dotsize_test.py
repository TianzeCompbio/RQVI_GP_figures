"""
Dot-size test for bottom-10 RQVI–Flashier pair exploration.

Picks the single worst pair (lowest best_corr) and plots a 2×4 UMAP grid
with 4 candidate dot sizes so the team can choose.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
from utils import (
    load_cluster_means, load_main_obs, load_umap_coords,
    FIG_DIR, PROJECT_DIR, PATH_RQVI_H5AD,
    LEVEL1_ORDER, CLUSTER_COL,
)

PATH_FLASHIER_CELL = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/cell_factor_matrix.txt"

DOT_SIZES = [0.5, 1.0, 2.0, 4.0]

# ─── Load bottom-1 pair ──────────────────────────────────────────────────────
best_corr_df = pd.read_csv(PROJECT_DIR / "data" / "cross_method_best_corr.csv")
best_corr_df = best_corr_df.sort_values("best_corr").reset_index(drop=True)
worst = best_corr_df.iloc[0]
flashier_factor = worst["flashier_factor"]       # e.g. "F51"
rqvi_gp = int(worst["best_rqvi_gp"])
corr_val = worst["best_corr"]
print(f"Worst pair: {flashier_factor} vs GP {rqvi_gp} (r={corr_val:.4f})")

# ─── Step 1: metadata ────────────────────────────────────────────────────────
print("Loading metadata...")
obs = load_main_obs()

# ─── Step 2: Flashier cell loadings ──────────────────────────────────────────
print("Loading Flashier cell loadings...")
flashier_cell = pd.read_csv(PATH_FLASHIER_CELL, sep="\t", index_col=0)
common_cells = flashier_cell.index.intersection(obs.index)
print(f"  Common cells: {len(common_cells)}")

# ─── Step 3: RQVI loadings ──────────────────────────────────────────────────
print("Loading RQVI h5ad...")
adata_rqvi = sc.read_h5ad(PATH_RQVI_H5AD)

# ─── Step 4: UMAP coords ────────────────────────────────────────────────────
print("Loading UMAP coords...")
umap_df = load_umap_coords()

common_all = (
    obs.index
    .intersection(umap_df.index)
    .intersection(pd.Index(adata_rqvi.obs_names))
    .intersection(flashier_cell.index)
)
print(f"  Common cells across all sources: {len(common_all)}")

# ─── Step 5: Downsample (stratified by level1) ──────────────────────────────
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

# RQVI loading for this GP
rqvi_idx_sampled = pd.Index(adata_rqvi.obs_names).get_indexer(sampled_idx)
vals_rqvi = adata_rqvi.X[rqvi_idx_sampled, rqvi_gp]
if hasattr(vals_rqvi, 'toarray'):
    vals_rqvi = vals_rqvi.toarray().ravel()
else:
    vals_rqvi = np.asarray(vals_rqvi).ravel()

# Flashier loading for this factor
vals_flashier = flashier_cell.loc[sampled_idx, flashier_factor].values

del adata_rqvi, flashier_cell
gc.collect()

shuffle_order = np.random.permutation(len(sampled_idx))

# ─── Plot 2×4 grid ──────────────────────────────────────────────────────────
print("Plotting...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for col_i, s_val in enumerate(DOT_SIZES):
    # Row 0: RQVI
    ax = axes[0, col_i]
    vals = vals_rqvi[shuffle_order]
    vmax = np.percentile(vals[vals > 0], 99) if (vals > 0).any() else 1.0
    ax.scatter(
        umap_coords[shuffle_order, 0], umap_coords[shuffle_order, 1],
        s=s_val, c=vals, cmap="viridis", vmin=0, vmax=vmax,
        alpha=0.6, rasterized=True,
    )
    ax.set_title(f"s={s_val}", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if col_i == 0:
        ax.set_ylabel(f"RQVI GP {rqvi_gp}", fontsize=10, fontweight="bold")

    # Row 1: Flashier
    ax = axes[1, col_i]
    vals = vals_flashier[shuffle_order]
    vmax = np.percentile(vals[vals > 0], 99) if (vals > 0).any() else 1.0
    ax.scatter(
        umap_coords[shuffle_order, 0], umap_coords[shuffle_order, 1],
        s=s_val, c=vals, cmap="viridis", vmin=0, vmax=vmax,
        alpha=0.6, rasterized=True,
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if col_i == 0:
        ax.set_ylabel(f"Flashier {flashier_factor}", fontsize=10, fontweight="bold")

fig.suptitle(
    f"Dot size test — worst pair: GP {rqvi_gp} vs {flashier_factor} (r={corr_val:.3f})",
    fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.95])

outpath = FIG_DIR / "bottom10_dotsize_test.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=200)
print(f"Saved to {outpath}")
plt.close(fig)
print("Done!")
