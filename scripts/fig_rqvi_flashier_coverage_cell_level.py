"""
Figure: RQVI coverage of Flashier GPs — cell-level correlation.

Same as fig_rqvi_flashier_coverage.py but computes Pearson correlation
directly across ~633k cells (no cluster aggregation). RQVI loadings are
loaded from h5ad files one seed at a time to limit memory usage.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from utils import FIG_DIR, load_main_obs

# ─── Paths ───────────────────────────────────────────────────────────────────
PATH_FLASHIER_CELL = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/cell_factor_matrix.txt"
PATH_RQVI_H5AD_PATTERN = "/data/tianzew/immgenT/RQVI_multiseeds/results/cmtloss08_64by4GPs_seed{seed}.h5ad"

N_SEEDS = 10
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ─── Step 1: Load Flashier cell loadings (no aggregation) ────────────────────
print("Loading cell obs for cell intersection...")
obs = load_main_obs()

print("Loading Flashier cell loadings (this may take a minute)...")
flashier_cell = pd.read_csv(PATH_FLASHIER_CELL, sep="\t", index_col=0)
print(f"  Flashier cell matrix: {flashier_cell.shape}")

# Start with cells present in Flashier and main obs
common_cells = flashier_cell.index.intersection(obs.index)
print(f"  Common cells (Flashier ∩ obs): {len(common_cells)}")

# Further intersect with all 10 RQVI h5ads to ensure consistent cell set
print("Intersecting cells across all RQVI seeds...")
for seed in range(N_SEEDS):
    path = PATH_RQVI_H5AD_PATTERN.format(seed=seed)
    adata = sc.read_h5ad(path, backed="r")
    common_cells = common_cells.intersection(adata.obs_names)
    adata.file.close()
    print(f"  After seed {seed}: {len(common_cells)} cells")

common_cells = common_cells.sort_values()
print(f"  Final common cells: {len(common_cells)}")

# Subset Flashier to common cells
flashier_mat = flashier_cell.loc[common_cells].values  # (n_cells, 200)
n_flashier_gps = flashier_mat.shape[1]
n_cells = flashier_mat.shape[0]
del flashier_cell
gc.collect()

# Pre-compute Flashier Z-scores (reused across all seeds)
print("Z-scoring Flashier cell loadings...")
F_z = (flashier_mat - flashier_mat.mean(axis=0, keepdims=True)) / (flashier_mat.std(axis=0, keepdims=True) + 1e-12)
del flashier_mat
gc.collect()

# ─── Step 2: Compute best-match correlations (one seed at a time) ────────────
print("Computing cell-level correlations (per seed)...")
best_corr = np.zeros((N_SEEDS, n_flashier_gps))

for s in range(N_SEEDS):
    path = PATH_RQVI_H5AD_PATTERN.format(seed=s)
    print(f"  Loading seed {s}...")
    adata = sc.read_h5ad(path)

    # Subset to common cells
    cell_idx = pd.Index(adata.obs_names).get_indexer(common_cells)
    R = adata.X[cell_idx, :]
    if hasattr(R, 'toarray'):
        R = R.toarray()
    R = R.astype(np.float32)

    del adata
    gc.collect()

    # Z-score RQVI across cells
    R_z = (R - R.mean(axis=0, keepdims=True)) / (R.std(axis=0, keepdims=True) + 1e-12)
    del R
    gc.collect()

    # Correlation matrix: (n_flashier_gps, n_rqvi_gps)
    corr_mat = (F_z.T @ R_z) / n_cells
    best_corr[s] = np.max(np.abs(corr_mat), axis=1)
    print(f"    Seed {s}: median best |r| = {np.median(best_corr[s]):.3f}")

    del R_z, corr_mat
    gc.collect()

del F_z
gc.collect()

# ─── Step 3: Coverage via greedy seed selection ──────────────────────────────
print("Computing coverage via greedy seed selection...")
coverage = np.zeros((N_SEEDS, len(THRESHOLDS)))

for ti, thresh in enumerate(THRESHOLDS):
    covered = np.zeros(n_flashier_gps, dtype=bool)
    remaining = set(range(N_SEEDS))
    for n in range(N_SEEDS):
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

# ─── Step 4: Plot ────────────────────────────────────────────────────────────
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
ax.set_title("RQVI Coverage of Flashier GPs vs. Number of Seeds (Cell-Level)",
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xlim(0.5, N_SEEDS + 0.5)
ymax = coverage.max()
ax.set_ylim(-0.02, min(ymax * 1.15, 1.05))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend outside
ax.legend(fontsize=8, title="Threshold", title_fontsize=9,
          bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

plt.tight_layout()
outpath = FIG_DIR / "rqvi_flashier_coverage_cell_level.pdf"
plt.savefig(outpath, bbox_inches="tight", dpi=200)
print(f"Saved to {outpath}")
