"""
GP sparsity scatter plot.

X-axis: proportion of activated cells (loading > 0.01)
Y-axis: proportion of activated genes (MAD-based outliers)
Color:  proportion of variance explained (PVE)

Saves:
  - data/gp_sparsity_scatter_data.csv   (intermediate, 256 rows)
  - figures/gp_sparsity_scatter.pdf     (scatter plot)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import median_abs_deviation

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import PROJECT_DIR, FIG_DIR, PATH_RQVI_H5AD, load_gene_effects

import scanpy as sc


# ─── 1. Fraction of activated cells ──────────────────────────────────────────
summary = pd.read_csv(PROJECT_DIR / "data/gp_summary_stats_seed0.csv")
frac_active_cells = summary["frac_active"].values  # length 256


# ─── 2. Fraction of activated genes (MAD-based outlier detection) ────────────
gene_effects = load_gene_effects()  # (19805 genes × 256 GPs)
W = gene_effects.values  # genes × GPs
n_genes = W.shape[0]

frac_active_genes = np.empty(W.shape[1])
for j in range(W.shape[1]):
    col = W[:, j]
    med = np.median(col)
    mad = median_abs_deviation(col)
    if mad == 0:
        # Fallback: no variation detected
        frac_active_genes[j] = 0.0
    else:
        modified_z = (col - med) / (mad * 1.4826)
        activated = np.abs(modified_z) > 3.5
        frac_active_genes[j] = activated.sum() / n_genes


# ─── 3. PVE (proportion of variance explained) ──────────────────────────────
print("Loading cell loadings h5ad …")
adata = sc.read_h5ad(PATH_RQVI_H5AD)
X = adata.X  # (cells × 256)
if hasattr(X, "toarray"):
    X = X.toarray()
X = np.asarray(X, dtype=np.float64)

# var_j = sum(X[:,j]^2) * sum(W[:,j]^2)
sum_x2 = (X ** 2).sum(axis=0)   # length 256
sum_w2 = (W ** 2).sum(axis=0)   # length 256
var_per_gp = sum_x2 * sum_w2
pve = var_per_gp / var_per_gp.sum()

del adata, X  # free memory


# ─── 4. Save intermediate data ──────────────────────────────────────────────
out_df = pd.DataFrame({
    "gp_idx": np.arange(256),
    "frac_active_cells": frac_active_cells,
    "frac_active_genes": frac_active_genes,
    "pve": pve,
})
out_path = PROJECT_DIR / "data/gp_sparsity_scatter_data.csv"
out_df.to_csv(out_path, index=False)
print(f"Saved {out_path}  ({len(out_df)} rows)")


# ─── 5. Scatter plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))

pve_pct = pve * 100
vmin = pve_pct[pve_pct > 0].min()
vmax = pve_pct.max()

# Clip zero values to a small positive number so they appear on log-scale x-axis
plot_frac_active_cells = frac_active_cells.copy()
min_pos = plot_frac_active_cells[plot_frac_active_cells > 0].min()
plot_frac_active_cells[plot_frac_active_cells == 0] = min_pos / 2

sc_plot = ax.scatter(
    plot_frac_active_cells,
    frac_active_genes,
    c=pve_pct,
    cmap="viridis",
    norm=LogNorm(vmin=vmin, vmax=vmax),
    s=28,
    edgecolors="white",
    linewidths=0.3,
    alpha=0.85,
)

cbar = fig.colorbar(sc_plot, ax=ax, pad=0.02)
cbar.set_label("% Variance Explained", fontsize=10)

ax.set_xscale("log")
ax.set_xlabel("Proportion of activated cells", fontsize=11)
ax.set_ylabel("Proportion of activated genes", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

fig.tight_layout()
fig_path = FIG_DIR / "gp_sparsity_scatter.pdf"
fig.savefig(fig_path, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved {fig_path}")
