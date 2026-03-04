"""
Panel D–F: RQVI vs Flashier factor pair comparison.

Two user-selected factor pairs with UMAP, MD plot, and dual-level
similarity histograms.

Pairs:
  1. RQVI GP 38 vs Flashier F58
  2. RQVI GP 45 vs Flashier F35
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy import sparse
from utils import (
    load_cluster_means, load_main_obs, load_gene_effects, load_umap_coords,
    FIG_DIR, PROJECT_DIR, PATH_RQVI_H5AD, PATH_MAIN_H5AD,
    LEVEL1_ORDER, CLUSTER_COL,
)

# ─── Config ───────────────────────────────────────────────────────────────────
PAIRS = [
    {"rqvi_gp": 38, "flashier_factor": 58, "color": "red", "label": "GP38 vs F58"},
    {"rqvi_gp": 45, "flashier_factor": 35, "color": "#1f77b4", "label": "GP45 vs F35"},
]

PATH_FLASHIER_CELL = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/cell_factor_matrix.txt"
PATH_FLASHIER_GENE = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/gene_factor_matrix.txt"

# ─── Step 1: Load metadata & RQVI cluster means ──────────────────────────────
print("Loading metadata and RQVI cluster means...")
obs = load_main_obs()
cluster_means_rqvi = load_cluster_means()  # 114 clusters x 256 GPs
print(f"  RQVI cluster means: {cluster_means_rqvi.shape}")

# ─── Step 2: Load Flashier cell loadings & compute cluster means ──────────────
print("Loading Flashier cell loadings...")
flashier_cell = pd.read_csv(PATH_FLASHIER_CELL, sep="\t", index_col=0)
print(f"  Flashier cell matrix: {flashier_cell.shape}")

common_cells = flashier_cell.index.intersection(obs.index)
print(f"  Common cells: {len(common_cells)}")

flashier_sub = flashier_cell.loc[common_cells].copy()
flashier_sub["cluster"] = obs.loc[common_cells, CLUSTER_COL].values
cluster_means_flashier = (
    flashier_sub.drop(columns=["cluster"])
    .groupby(flashier_sub["cluster"])
    .mean()
)
print(f"  Flashier cluster means: {cluster_means_flashier.shape}")

# Keep only clusters present in both
common_clusters = cluster_means_rqvi.index.intersection(cluster_means_flashier.index)
cluster_means_rqvi_common = cluster_means_rqvi.loc[common_clusters]
cluster_means_flashier_common = cluster_means_flashier.loc[common_clusters]

# ─── Step 3: Compute pair-specific correlations (cluster-level) ───────────────
print("Computing cluster-level correlations for selected pairs...")
R = cluster_means_rqvi_common.values
F = cluster_means_flashier_common.values
R_z = (R - R.mean(axis=0, keepdims=True)) / (R.std(axis=0, keepdims=True) + 1e-12)
F_z = (F - F.mean(axis=0, keepdims=True)) / (F.std(axis=0, keepdims=True) + 1e-12)
corr_matrix_cluster = (R_z.T @ F_z) / R_z.shape[0]  # (256, 200)

rqvi_col_names = cluster_means_rqvi_common.columns.tolist()
flashier_col_names = cluster_means_flashier_common.columns.tolist()

for p in PAIRS:
    ri = rqvi_col_names.index(str(p["rqvi_gp"]))
    fi = flashier_col_names.index(f"F{p['flashier_factor']}")
    p["corr_cluster"] = corr_matrix_cluster[ri, fi]
    print(f"  {p['label']}: cluster-level r = {p['corr_cluster']:.4f}")

# ─── Step 4: Cell-level correlations (memory efficient, 2 pairs only) ─────────
print("Computing cell-level correlations...")
adata_rqvi = sc.read_h5ad(PATH_RQVI_H5AD)
common_cells_both = common_cells.intersection(pd.Index(adata_rqvi.obs_names))
print(f"  Common cells for cell-level: {len(common_cells_both)}")

rqvi_idx = pd.Index(adata_rqvi.obs_names).get_indexer(common_cells_both)

for p in PAIRS:
    gp_col = p["rqvi_gp"]
    fl_col = f"F{p['flashier_factor']}"

    r_vals = adata_rqvi.X[rqvi_idx, gp_col]
    if hasattr(r_vals, 'toarray'):
        r_vals = r_vals.toarray().ravel()
    else:
        r_vals = np.asarray(r_vals).ravel()
    f_vals = flashier_cell.loc[common_cells_both, fl_col].values

    r_z = (r_vals - r_vals.mean()) / (r_vals.std() + 1e-12)
    f_z = (f_vals - f_vals.mean()) / (f_vals.std() + 1e-12)
    p["corr_cell"] = float(np.dot(r_z, f_z) / len(r_z))
    print(f"  {p['label']}: cell-level r = {p['corr_cell']:.4f}")

# ─── Step 5: Prepare UMAP data ───────────────────────────────────────────────
print("Preparing UMAP data...")
umap_df = load_umap_coords()

common_all = (
    obs.index
    .intersection(umap_df.index)
    .intersection(pd.Index(adata_rqvi.obs_names))
    .intersection(flashier_cell.index)
)
print(f"  Common cells across all sources: {len(common_all)}")

# Downsample stratified by level1
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

# Get RQVI loadings for the 2 GPs
rqvi_idx_sampled = pd.Index(adata_rqvi.obs_names).get_indexer(sampled_idx)
rqvi_loadings = {}
for p in PAIRS:
    vals = adata_rqvi.X[rqvi_idx_sampled, p["rqvi_gp"]]
    if hasattr(vals, 'toarray'):
        vals = vals.toarray().ravel()
    else:
        vals = np.asarray(vals).ravel()
    rqvi_loadings[p["rqvi_gp"]] = vals

# Get Flashier loadings for the 2 factors
flashier_loadings = {}
for p in PAIRS:
    col = f"F{p['flashier_factor']}"
    flashier_loadings[p["flashier_factor"]] = flashier_cell.loc[sampled_idx, col].values

del adata_rqvi, flashier_cell, flashier_sub
gc.collect()

shuffle_order = np.random.permutation(len(sampled_idx))

# ─── Step 6: Load gene effects & compute mean expression for MD plots ─────────
print("Loading gene effects...")
rqvi_effects = load_gene_effects()  # 19805 genes x 256 GPs, cols "0"-"255"
print(f"  RQVI gene effects: {rqvi_effects.shape}")

flashier_gene = pd.read_csv(PATH_FLASHIER_GENE, sep="\t", index_col=0)
print(f"  Flashier gene effects: {flashier_gene.shape}")

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

# Intersect genes across all three sources
common_genes = (
    rqvi_effects.index
    .intersection(flashier_gene.index)
    .intersection(mean_log_expr.index)
)
print(f"  Common genes: {len(common_genes)}")
rqvi_effects = rqvi_effects.loc[common_genes]
flashier_gene = flashier_gene.loc[common_genes]
mean_expr = mean_log_expr.loc[common_genes]

# ─── Step 7: Plot ─────────────────────────────────────────────────────────────
print("Plotting...")
import matplotlib
matplotlib.use("Agg")

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(
    3, 4,
    height_ratios=[2.5, 3.0, 2.5],
    hspace=0.35, wspace=0.30,
)

# ── Row 0: UMAP panels ──────────────────────────────────────────────────────
for pair_i, p in enumerate(PAIRS):
    col_offset = pair_i * 2

    # RQVI UMAP
    ax = fig.add_subplot(gs[0, col_offset])
    vals = rqvi_loadings[p["rqvi_gp"]][shuffle_order]
    vmax = np.percentile(vals[vals > 0], 99) if (vals > 0).any() else 1.0
    ax.scatter(
        umap_coords[shuffle_order, 0], umap_coords[shuffle_order, 1],
        s=0.3, c=vals, cmap="viridis", vmin=0, vmax=vmax,
        alpha=0.6, rasterized=True,
    )
    ax.set_title(f"RQVI GP {p['rqvi_gp']}", fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if pair_i == 0:
        ax.set_ylabel("UMAP", fontsize=10, fontweight="bold")

    # Flashier UMAP
    ax = fig.add_subplot(gs[0, col_offset + 1])
    vals = flashier_loadings[p["flashier_factor"]][shuffle_order]
    vmax = np.percentile(vals[vals > 0], 99) if (vals > 0).any() else 1.0
    ax.scatter(
        umap_coords[shuffle_order, 0], umap_coords[shuffle_order, 1],
        s=0.3, c=vals, cmap="viridis", vmin=0, vmax=vmax,
        alpha=0.6, rasterized=True,
    )
    ax.set_title(f"Flashier F{p['flashier_factor']}", fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# ── Row 1: MD plots ─────────────────────────────────────────────────────────
for pair_i, p in enumerate(PAIRS):
    col_offset = pair_i * 2

    for sub_i, (method, effects_df, gp_key) in enumerate([
        ("RQVI", rqvi_effects, str(p["rqvi_gp"])),
        ("Flashier", flashier_gene, f"V{p['flashier_factor']}"),
    ]):
        ax = fig.add_subplot(gs[1, col_offset + sub_i])
        gene_weights = effects_df[gp_key]

        df = pd.DataFrame({"x": gene_weights, "y": mean_expr}).dropna()

        # Grey cloud
        ax.scatter(df["x"], df["y"], s=8, c="0.85", alpha=0.2,
                   linewidths=0, zorder=1)

        # Top 15 genes by absolute weight
        top15 = df["x"].abs().nlargest(15).index
        hl_df = df.loc[top15]
        ax.scatter(hl_df["x"], hl_df["y"], s=20, facecolors="none",
                   edgecolors="#1f77b4", linewidths=0.9, zorder=3)

        # Label top 8
        top8 = hl_df["x"].abs().nlargest(8).index
        for gene in top8:
            ax.annotate(
                gene, (hl_df.loc[gene, "x"], hl_df.loc[gene, "y"]),
                fontsize=5.5, color="#1f77b4", alpha=0.9,
                xytext=(3, 3), textcoords="offset points",
            )

        if method == "RQVI":
            title = f"RQVI GP {p['rqvi_gp']}"
        else:
            title = f"Flashier F{p['flashier_factor']}"
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.set_xlabel("Gene effect", fontsize=8)
        if sub_i == 0 and pair_i == 0:
            ax.set_ylabel("Mean log expr", fontsize=8)
        else:
            ax.set_ylabel("")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

# ── Row 2: Histograms ───────────────────────────────────────────────────────
# Load pre-computed best-match correlation tables
best_corr_cluster_df = pd.read_csv(PROJECT_DIR / "data" / "cross_method_best_corr.csv")
best_corr_cell_df = pd.read_csv(PROJECT_DIR / "data" / "cross_method_best_corr_cell_level.csv")

# Cluster-level histogram (left half)
ax_hist_cl = fig.add_subplot(gs[2, 0:2])
ax_hist_cl.hist(best_corr_cluster_df["best_corr"], bins=30,
                color="#4C72B0", edgecolor="white", linewidth=0.5, alpha=0.7)
for p in PAIRS:
    ax_hist_cl.axvline(p["corr_cluster"], color=p["color"], linestyle="--",
                       linewidth=1.5, label=f"{p['label']} (r={p['corr_cluster']:.2f})")
ax_hist_cl.set_xlabel("Best-match Pearson r (cluster-level)", fontsize=9)
ax_hist_cl.set_ylabel("Count (Flashier factors)", fontsize=9)
ax_hist_cl.set_title("Cluster-level similarity (n=200)", fontsize=10,
                     fontweight="bold", loc="left")
ax_hist_cl.legend(fontsize=7, loc="upper left")
ax_hist_cl.spines["top"].set_visible(False)
ax_hist_cl.spines["right"].set_visible(False)

# Cell-level histogram (right half)
ax_hist_cell = fig.add_subplot(gs[2, 2:4])
ax_hist_cell.hist(best_corr_cell_df["best_corr"], bins=30,
                  color="#4C72B0", edgecolor="white", linewidth=0.5, alpha=0.7)
for p in PAIRS:
    ax_hist_cell.axvline(p["corr_cell"], color=p["color"], linestyle="--",
                         linewidth=1.5, label=f"{p['label']} (r={p['corr_cell']:.2f})")
ax_hist_cell.set_xlabel("Best-match Pearson r (cell-level)", fontsize=9)
ax_hist_cell.set_ylabel("Count (Flashier factors)", fontsize=9)
ax_hist_cell.set_title("Cell-level similarity (n=200)", fontsize=10,
                       fontweight="bold", loc="left")
ax_hist_cell.legend(fontsize=7, loc="upper left")
ax_hist_cell.spines["top"].set_visible(False)
ax_hist_cell.spines["right"].set_visible(False)

# ─── Save ─────────────────────────────────────────────────────────────────────
outpath = FIG_DIR / "panel_D_F.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=200)
print(f"Saved figure to {outpath}")
plt.close(fig)

# Save pair correlations CSV
pair_rows = []
for p in PAIRS:
    pair_rows.append({
        "rqvi_gp": p["rqvi_gp"],
        "flashier_factor": p["flashier_factor"],
        "corr_cluster": p["corr_cluster"],
        "corr_cell": p["corr_cell"],
    })
csv_out = PROJECT_DIR / "data" / "panel_D_F_pair_correlations.csv"
pd.DataFrame(pair_rows).to_csv(csv_out, index=False)
print(f"Saved pair correlations to {csv_out}")

print("Done!")
