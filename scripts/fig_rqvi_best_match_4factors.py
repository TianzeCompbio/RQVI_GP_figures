"""
Best-matched RQVI GPs for Flashier F22, F30, F58, F68.

Searches across 10 seeds x 256 GPs to find the genuine best match for each
Flashier factor, then plots UMAP (top row) and MD (bottom row) for the RQVI
side only, using a purple colormap.
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
from scipy import sparse
from matplotlib.colors import LinearSegmentedColormap
from utils import (
    load_main_obs, md_scatter,
    FIG_DIR, PATH_MAIN_H5AD,
    LEVEL1_ORDER, CLUSTER_COL,
)

# ─── Quick test toggle ───────────────────────────────────────────────────────
QUICK_TEST = False  # True: only F58, 20k cells; False: all 4 factors, 100k cells

# ─── Config ───────────────────────────────────────────────────────────────────
TARGET_FACTORS = [58] if QUICK_TEST else [22, 30, 58, 68]
TARGET_N = 20_000 if QUICK_TEST else 100_000

PATH_FLASHIER_CELL = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/cell_factor_matrix.txt"
CORR_RST_DIR = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/function_analysis/corr_rst"
PATH_RQVI_UMAP_H5AD = "/data/tianzew/immgenT/RQVI/cmtloss08_64by4GPs_mde_totalVI.h5ad"
RQVI_CELL_H5AD_TEMPLATE = "/data/tianzew/immgenT/RQVI_multiseeds/results/cmtloss08_64by4GPs_seed{}.h5ad"
RQVI_GENE_EFFECTS_TEMPLATE = "/homes/gws/tianzew/projects/GP_figures/data/RQVI_gene_factors/gp_effects_matrix_seed{}.csv"
N_SEEDS = 10

# Purple colormap
purples_cmap = LinearSegmentedColormap.from_list(
    "custom_purples", ["#f0f0f0", "#c6b3d9", "#7b5ea7", "#3f2d76"], N=256)

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Best-match computation
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: Finding best-matched RQVI GPs across 10 seeds")
print("=" * 60)

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

print("Loading 10 RQVI cluster means CSVs...")
rqvi_cluster_means_list = []
for seed in range(N_SEEDS):
    path = f"{CORR_RST_DIR}/rqvi_seed{seed}_gp_cell_level.csv"
    df = pd.read_csv(path, index_col="group")
    df.columns = df.columns.astype(str)
    rqvi_cluster_means_list.append(df)

# Find common clusters
common_clusters = flashier_cluster_means.index
for df in rqvi_cluster_means_list:
    common_clusters = common_clusters.intersection(df.index)
common_clusters = sorted(common_clusters)
print(f"  Common clusters: {len(common_clusters)}")

# For each target factor, find best (seed, GP, correlation)
F_mat = flashier_cluster_means.loc[common_clusters].values  # (n_clusters, n_flashier)
flashier_cols = flashier_cluster_means.columns.tolist()

best_matches = {}  # factor -> {"seed": int, "gp": int, "corr": float}

for f_idx_target in TARGET_FACTORS:
    # Find column index for this Flashier factor (columns are F1, F2, ...)
    col_name = f"F{f_idx_target}"
    f_col_idx = flashier_cols.index(col_name)
    f_vec = F_mat[:, f_col_idx]
    f_z = (f_vec - f_vec.mean()) / (f_vec.std() + 1e-12)

    best_seed, best_gp, best_corr = -1, -1, -np.inf

    for s in range(N_SEEDS):
        R_mat = rqvi_cluster_means_list[s].loc[common_clusters].values
        R_z = (R_mat - R_mat.mean(axis=0, keepdims=True)) / (R_mat.std(axis=0, keepdims=True) + 1e-12)
        # Pearson r between f_z and each RQVI GP
        corrs = (f_z @ R_z) / len(f_z)
        gp_idx = int(np.argmax(corrs))
        r_val = corrs[gp_idx]
        if r_val > best_corr:
            best_seed, best_gp, best_corr = s, gp_idx, r_val

    best_matches[f_idx_target] = {"seed": best_seed, "gp": best_gp, "corr": best_corr}

del rqvi_cluster_means_list

print("\nBest matches:")
print(f"{'Flashier F':>12s} {'Seed':>6s} {'RQVI GP':>8s} {'Pearson r':>10s}")
print("-" * 40)
for f_idx in TARGET_FACTORS:
    m = best_matches[f_idx]
    print(f"{'F' + str(f_idx):>12s} {m['seed']:>6d} {m['gp']:>8d} {m['corr']:>10.4f}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: UMAP data loading & downsampling
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 2: Loading UMAP data and downsampling")
print("=" * 60)

print("Loading UMAP coordinates...")
adata_umap = sc.read_h5ad(PATH_RQVI_UMAP_H5AD)

# Intersect cells (only need obs and UMAP, no Flashier)
common_all = obs.index.intersection(pd.Index(adata_umap.obs_names))
print(f"  Common cells (obs & UMAP): {len(common_all)}")

# Stratified downsample
np.random.seed(42)
obs_sub = obs.loc[common_all]
sampled_idx = []
for level1 in LEVEL1_ORDER:
    cells = obs_sub[obs_sub["level1"] == level1].index
    n_sample = max(1, int(len(cells) / len(common_all) * TARGET_N))
    n_sample = min(n_sample, len(cells))
    sampled_idx.extend(np.random.choice(cells, n_sample, replace=False))
sampled_idx = np.array(sampled_idx)
print(f"  Downsampled to {len(sampled_idx)} cells")

# Build subset AnnData
adata_sub = adata_umap[sampled_idx].copy()
del adata_umap
gc.collect()

# Load RQVI cell loadings for each unique seed needed
unique_seeds = set(m["seed"] for m in best_matches.values())
print(f"  Loading cell loadings for seeds: {sorted(unique_seeds)}")

for seed in unique_seeds:
    path = RQVI_CELL_H5AD_TEMPLATE.format(seed)
    print(f"  Loading seed {seed} from {path}...")
    adata_rqvi = sc.read_h5ad(path)
    rqvi_idx = pd.Index(adata_rqvi.obs_names).get_indexer(sampled_idx)

    # Extract GP columns needed from this seed
    for f_idx, m in best_matches.items():
        if m["seed"] != seed:
            continue
        gp = m["gp"]
        vals = adata_rqvi.X[rqvi_idx, gp]
        if hasattr(vals, "toarray"):
            vals = vals.toarray().ravel()
        else:
            vals = np.asarray(vals).ravel()
        col_name = f"rqvi_s{seed}_gp{gp}"
        adata_sub.obs[col_name] = vals
        print(f"    Extracted GP {gp} -> {col_name}")

    del adata_rqvi
    gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Mean expression computation
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 3: Computing mean log expression per gene")
print("=" * 60)

adata_main = sc.read_h5ad(PATH_MAIN_H5AD, backed="r")
n_cells = adata_main.shape[0]
n_genes = adata_main.shape[1]
gene_names = adata_main.var_names.tolist()
chunk_size = 50_000
gene_sums = np.zeros(n_genes, dtype=np.float64)

for start in range(0, n_cells, chunk_size):
    end = min(start + chunk_size, n_cells)
    chunk = adata_main.X[start:end]
    if sparse.issparse(chunk):
        chunk = chunk.toarray()
    gene_sums += chunk.sum(axis=0).ravel()
    if start % 200_000 == 0:
        print(f"  Processed {start}/{n_cells} cells...")

mean_log_expr = pd.Series(gene_sums / n_cells, index=gene_names)
adata_main.file.close()
print(f"  Mean expression computed for {len(mean_log_expr)} genes")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Gene effects loading
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 4: Loading gene effects")
print("=" * 60)

gene_effects_by_factor = {}  # f_idx -> Series

for seed in unique_seeds:
    path = RQVI_GENE_EFFECTS_TEMPLATE.format(seed)
    print(f"  Loading gene effects for seed {seed}...")
    effects_df = pd.read_csv(path, index_col=0)
    effects_df.columns = effects_df.columns.astype(str)

    for f_idx, m in best_matches.items():
        if m["seed"] != seed:
            continue
        gp = m["gp"]
        gp_col = str(gp)
        gene_effects_by_factor[f_idx] = effects_df[gp_col]
        print(f"    F{f_idx} -> seed {seed}, GP {gp}")

    del effects_df
    gc.collect()

# Intersect genes with mean_log_expr
common_genes = mean_log_expr.index
for f_idx in TARGET_FACTORS:
    common_genes = common_genes.intersection(gene_effects_by_factor[f_idx].index)
print(f"  Common genes across all: {len(common_genes)}")

mean_expr = mean_log_expr.loc[common_genes]
for f_idx in TARGET_FACTORS:
    gene_effects_by_factor[f_idx] = gene_effects_by_factor[f_idx].loc[common_genes]

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Plotting
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 5: Plotting")
print("=" * 60)

n_factors = len(TARGET_FACTORS)
fig, axes = plt.subplots(2, n_factors, figsize=(5 * n_factors, 8))

# Handle single-column case (QUICK_TEST)
if n_factors == 1:
    axes = axes.reshape(2, 1)

for col_i, f_idx in enumerate(TARGET_FACTORS):
    m = best_matches[f_idx]
    seed, gp, corr = m["seed"], m["gp"], m["corr"]
    obs_col = f"rqvi_s{seed}_gp{gp}"

    # --- Top row: UMAP ---
    ax = axes[0, col_i]
    vals = adata_sub.obs[obs_col].values
    vmax = np.percentile(vals[vals > 0], 98) if (vals > 0).any() else 1.0
    sc.pl.umap(adata_sub, color=obs_col, ax=ax, show=False,
               title=f"GP {gp} (seed {seed})",
               frameon=False, size=15,
               cmap=purples_cmap, vmin=0, vmax=vmax)

    # --- Bottom row: MD ---
    ax = axes[1, col_i]
    title = f"GP {gp} (seed {seed})"
    md_scatter(ax, gene_effects_by_factor[f_idx], mean_expr, title,
               point_size_bg=20, point_size_hl=50)
    if col_i == 0:
        ax.set_ylabel("Mean log expr", fontsize=11)
    else:
        ax.set_ylabel("")

    # Column label at top
    axes[0, col_i].text(
        0.5, 1.15, f"F{f_idx} match (r={corr:.3f})",
        transform=axes[0, col_i].transAxes,
        ha="center", fontsize=10, fontweight="bold")

fig.suptitle("Best-matched RQVI GPs for Flashier factors", fontsize=13,
             fontweight="bold", y=1.02)
fig.tight_layout()

# ─── Save ─────────────────────────────────────────────────────────────────────
suffix = "_quicktest" if QUICK_TEST else ""
outpath = FIG_DIR / f"rqvi_best_match_4factors{suffix}.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=200)
print(f"\nSaved figure to {outpath}")
plt.close(fig)
print("Done!")
