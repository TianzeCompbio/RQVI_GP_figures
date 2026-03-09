"""
Standalone single-pair figures: one row of 4 panels
(RQVI UMAP, Flashier UMAP, RQVI MD, Flashier MD).

Produces one PDF per pair.
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
from utils import (
    load_cluster_means, load_main_obs, load_gene_effects, load_umap_coords,
    FIG_DIR, PROJECT_DIR, PATH_RQVI_H5AD, PATH_MAIN_H5AD,
    LEVEL1_ORDER, CLUSTER_COL,
)

PATH_FLASHIER_CELL = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/cell_factor_matrix.txt"
PATH_FLASHIER_GENE = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/gene_factor_matrix.txt"

DOT_SIZE = 1.0

# ─── Pairs to generate standalone figures for ────────────────────────────────
PAIRS = [
    {"flashier_factor": "F58", "rqvi_gp": 38, "best_corr": 0.573},
    {"flashier_factor": "F35", "rqvi_gp": 45, "best_corr": 0.436},
]

rqvi_gps = sorted(set(p["rqvi_gp"] for p in PAIRS))
flashier_factors = sorted(set(p["flashier_factor"] for p in PAIRS))

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

# ─── Step 4: UMAP coords & downsampling ─────────────────────────────────────
print("Loading UMAP coords...")
umap_df = load_umap_coords()

common_all = (
    obs.index
    .intersection(umap_df.index)
    .intersection(pd.Index(adata_rqvi.obs_names))
    .intersection(flashier_cell.index)
)
print(f"  Common cells across all sources: {len(common_all)}")

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

# ─── Step 5: Extract RQVI & Flashier loadings ───────────────────────────────
print("Extracting cell loadings for all pairs...")
rqvi_idx_sampled = pd.Index(adata_rqvi.obs_names).get_indexer(sampled_idx)

rqvi_loadings = {}
for gp in rqvi_gps:
    vals = adata_rqvi.X[rqvi_idx_sampled, gp]
    if hasattr(vals, 'toarray'):
        vals = vals.toarray().ravel()
    else:
        vals = np.asarray(vals).ravel()
    rqvi_loadings[gp] = vals

flashier_loadings = {}
for ff in flashier_factors:
    flashier_loadings[ff] = flashier_cell.loc[sampled_idx, ff].values

del adata_rqvi, flashier_cell
gc.collect()

shuffle_order = np.random.permutation(len(sampled_idx))

# ─── Step 6: Gene effects & mean expression for MD plots ────────────────────
print("Loading gene effects...")
rqvi_effects = load_gene_effects()
flashier_gene = pd.read_csv(PATH_FLASHIER_GENE, sep="\t", index_col=0)

print("Computing mean log expression per gene (chunked backed mode)...")
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

common_genes = (
    rqvi_effects.index
    .intersection(flashier_gene.index)
    .intersection(mean_log_expr.index)
)
print(f"  Common genes: {len(common_genes)}")
rqvi_effects = rqvi_effects.loc[common_genes]
flashier_gene = flashier_gene.loc[common_genes]
mean_expr = mean_log_expr.loc[common_genes]

# ─── Step 7: Plot one figure per pair ────────────────────────────────────────
for p in PAIRS:
    gp = p["rqvi_gp"]
    ff = p["flashier_factor"]
    ff_num = ff[1:]  # strip "F"
    corr = p["best_corr"]
    label = f"GP {gp} vs {ff} (r={corr:.3f})"
    print(f"Plotting {label}...")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    # Panel 0: RQVI UMAP
    ax = axes[0]
    vals = rqvi_loadings[gp][shuffle_order]
    vmax = np.percentile(vals[vals > 0], 99) if (vals > 0).any() else 1.0
    ax.scatter(
        umap_coords[shuffle_order, 0], umap_coords[shuffle_order, 1],
        s=DOT_SIZE, c=vals, cmap="viridis", vmin=0, vmax=vmax,
        alpha=0.6, rasterized=True,
    )
    ax.set_title(f"RQVI GP {gp}", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Panel 1: Flashier UMAP
    ax = axes[1]
    vals = flashier_loadings[ff][shuffle_order]
    vmax = np.percentile(vals[vals > 0], 99) if (vals > 0).any() else 1.0
    ax.scatter(
        umap_coords[shuffle_order, 0], umap_coords[shuffle_order, 1],
        s=DOT_SIZE, c=vals, cmap="viridis", vmin=0, vmax=vmax,
        alpha=0.6, rasterized=True,
    )
    ax.set_title(f"Flashier {ff}", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Panel 2: RQVI MD plot
    ax = axes[2]
    gp_col = str(gp)
    gene_weights = rqvi_effects[gp_col]
    df_md = pd.DataFrame({"x": gene_weights, "y": mean_expr}).dropna()
    ax.scatter(df_md["x"], df_md["y"], s=8, c="0.85", alpha=0.2,
               linewidths=0, zorder=1)
    top15 = df_md["x"].abs().nlargest(15).index
    hl_df = df_md.loc[top15]
    ax.scatter(hl_df["x"], hl_df["y"], s=20, facecolors="none",
               edgecolors="#1f77b4", linewidths=0.9, zorder=3)
    top8 = hl_df["x"].abs().nlargest(8).index
    for gene in top8:
        ax.annotate(
            gene, (hl_df.loc[gene, "x"], hl_df.loc[gene, "y"]),
            fontsize=6, color="#1f77b4", alpha=0.9,
            xytext=(3, 3), textcoords="offset points",
        )
    ax.set_title(f"RQVI GP {gp}", fontsize=10, fontweight="bold", loc="left")
    ax.set_xlabel("Gene effect", fontsize=8)
    ax.set_ylabel("Mean log expr", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Panel 3: Flashier MD plot
    ax = axes[3]
    fl_col = f"V{ff_num}"
    gene_weights = flashier_gene[fl_col]
    df_md = pd.DataFrame({"x": gene_weights, "y": mean_expr}).dropna()
    ax.scatter(df_md["x"], df_md["y"], s=8, c="0.85", alpha=0.2,
               linewidths=0, zorder=1)
    top15 = df_md["x"].abs().nlargest(15).index
    hl_df = df_md.loc[top15]
    ax.scatter(hl_df["x"], hl_df["y"], s=20, facecolors="none",
               edgecolors="#1f77b4", linewidths=0.9, zorder=3)
    top8 = hl_df["x"].abs().nlargest(8).index
    for gene in top8:
        ax.annotate(
            gene, (hl_df.loc[gene, "x"], hl_df.loc[gene, "y"]),
            fontsize=6, color="#1f77b4", alpha=0.9,
            xytext=(3, 3), textcoords="offset points",
        )
    ax.set_title(f"Flashier {ff}", fontsize=10, fontweight="bold", loc="left")
    ax.set_xlabel("Gene effect", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle(label, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    outpath = FIG_DIR / f"pair_GP{gp}_{ff}.pdf"
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    print(f"  Saved {outpath}")
    plt.close(fig)

print("Done!")
