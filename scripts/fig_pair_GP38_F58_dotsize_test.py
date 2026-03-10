"""
GP 38 vs Flashier F58: dot-size variations for UMAP panels.
Generates one PDF per dot size: 5, 10, 20, 40.
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
    load_main_obs, load_gene_effects,
    FIG_DIR, PATH_RQVI_H5AD, PATH_MAIN_H5AD,
    LEVEL1_ORDER,
)

PATH_RQVI_UMAP_H5AD = "/data/tianzew/immgenT/RQVI/cmtloss08_64by4GPs_mde_totalVI.h5ad"

# ─── Config ───────────────────────────────────────────────────────────────────
PAIR = {"rqvi_gp": 38, "flashier_factor": 58}
DOT_SIZES = [5, 10, 20, 40]

PATH_FLASHIER_CELL = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/cell_factor_matrix.txt"
PATH_FLASHIER_GENE = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/gene_factor_matrix.txt"

# ─── Load data (once) ───────────────────────────────────────────────────────
print("Loading metadata...")
obs = load_main_obs()

print("Loading Flashier cell loadings...")
flashier_cell = pd.read_csv(PATH_FLASHIER_CELL, sep="\t", index_col=0)

print("Loading RQVI cell loadings...")
adata_rqvi = sc.read_h5ad(PATH_RQVI_H5AD)

print("Loading UMAP coordinates...")
adata_umap = sc.read_h5ad(PATH_RQVI_UMAP_H5AD)

# ─── Intersect cells ─────────────────────────────────────────────────────────
common_all = (
    obs.index
    .intersection(pd.Index(adata_rqvi.obs_names))
    .intersection(pd.Index(adata_umap.obs_names))
    .intersection(flashier_cell.index)
)
print(f"  Common cells: {len(common_all)}")

# ─── Stratified downsample ──────────────────────────────────────────────────
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

# ─── Build subset AnnData ───────────────────────────────────────────────────
adata_sub = adata_umap[sampled_idx].copy()

gp = PAIR["rqvi_gp"]
rqvi_idx = pd.Index(adata_rqvi.obs_names).get_indexer(sampled_idx)
vals = adata_rqvi.X[rqvi_idx, gp]
if hasattr(vals, 'toarray'):
    vals = vals.toarray().ravel()
else:
    vals = np.asarray(vals).ravel()
adata_sub.obs[f"rqvi_gp{gp}"] = vals

col = f"F{PAIR['flashier_factor']}"
adata_sub.obs[f"flashier_{col}"] = flashier_cell.loc[sampled_idx, col].values

del adata_rqvi, adata_umap, flashier_cell
gc.collect()

# ─── Load gene effects for MD plots ─────────────────────────────────────────
print("Loading gene effects...")
rqvi_effects = load_gene_effects()
flashier_gene = pd.read_csv(PATH_FLASHIER_GENE, sep="\t", index_col=0)

# ─── Compute mean log expression ────────────────────────────────────────────
print("Computing mean log expression...")
adata_main = sc.read_h5ad(PATH_MAIN_H5AD, backed="r")
n_cells = adata_main.shape[0]
n_genes = adata_main.shape[1]
gene_names = adata_main.var_names.tolist()
gene_sums = np.zeros(n_genes, dtype=np.float64)

for start in range(0, n_cells, 50_000):
    end = min(start + 50_000, n_cells)
    chunk = adata_main.X[start:end]
    if sparse.issparse(chunk):
        chunk = chunk.toarray()
    gene_sums += chunk.sum(axis=0).ravel()
    if start % 200_000 == 0:
        print(f"  Processed {start}/{n_cells} cells...")

mean_log_expr = pd.Series(gene_sums / n_cells, index=gene_names)
adata_main.file.close()

# ─── Intersect genes ────────────────────────────────────────────────────────
common_genes = rqvi_effects.index.intersection(flashier_gene.index).intersection(mean_log_expr.index)
rqvi_effects = rqvi_effects.loc[common_genes]
flashier_gene = flashier_gene.loc[common_genes]
mean_expr = mean_log_expr.loc[common_genes]

# ─── Precompute Flashier vmax ────────────────────────────────────────────────
flashier_vals = adata_sub.obs[f"flashier_{col}"].values
flashier_vmax = np.percentile(flashier_vals[flashier_vals > 0], 98) if (flashier_vals > 0).any() else 1.0

# ─── Generate one figure per dot size ────────────────────────────────────────
for dot_size in DOT_SIZES:
    print(f"Plotting dot size = {dot_size}...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Top row: UMAP
    sc.pl.umap(adata_sub, color=f"rqvi_gp{gp}", ax=axes[0, 0],
               show=False, title=f"RQVI GP {gp}", frameon=False, size=dot_size)
    sc.pl.umap(adata_sub, color=f"flashier_{col}", ax=axes[0, 1],
               show=False, title=f"Flashier {col}", frameon=False, size=dot_size,
               vmax=flashier_vmax)

    # Bottom row: MD plots
    for sub_i, (method, effects_df, gp_key) in enumerate([
        ("RQVI", rqvi_effects, str(PAIR["rqvi_gp"])),
        ("Flashier", flashier_gene, f"V{PAIR['flashier_factor']}"),
    ]):
        ax = axes[1, sub_i]
        gene_weights = effects_df[gp_key]
        df = pd.DataFrame({"x": gene_weights, "y": mean_expr}).dropna()

        ax.scatter(df["x"], df["y"], s=8, c="0.85", alpha=0.2, linewidths=0, zorder=1)

        top15 = df["x"].abs().nlargest(15).index
        hl_df = df.loc[top15]
        ax.scatter(hl_df["x"], hl_df["y"], s=20, facecolors="none",
                   edgecolors="#1f77b4", linewidths=0.9, zorder=3)

        top8 = hl_df["x"].abs().nlargest(8).index
        for gene in top8:
            ax.annotate(gene, (hl_df.loc[gene, "x"], hl_df.loc[gene, "y"]),
                        fontsize=5.5, color="#1f77b4", alpha=0.9,
                        xytext=(3, 3), textcoords="offset points")

        title = f"RQVI GP {PAIR['rqvi_gp']}" if method == "RQVI" else f"Flashier F{PAIR['flashier_factor']}"
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.set_xlabel("Gene effect", fontsize=8)
        ax.set_ylabel("Mean log expr" if sub_i == 0 else "", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle(f"GP 38 vs F58 (r = 0.573) — dot size {dot_size}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    outpath = FIG_DIR / f"pair_GP38_F58_dots{dot_size}.pdf"
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    print(f"  Saved {outpath}")
    plt.close(fig)

print("Done!")
