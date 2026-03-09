"""
Bottom 10 RQVI–Flashier pairs: UMAP + MD plots for each pair, plus
a cluster-level similarity histogram marking all 10 pairs.

Layout: gridspec(11, 4)
  - Rows 0–9: one pair each → [RQVI UMAP, Flashier UMAP, RQVI MD, Flashier MD]
  - Row 10: histogram spanning all 4 columns
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

# User-selected dot size from Phase 1 (update after testing)
DOT_SIZE = 1.0

# ─── Load bottom 10 pairs ───────────────────────────────────────────────────
best_corr_df = pd.read_csv(PROJECT_DIR / "data" / "cross_method_best_corr.csv")
best_corr_df = best_corr_df.sort_values("best_corr").reset_index(drop=True)

# Filter to robust RQVI GPs (reproduced in >= 5 of 9 other seeds)
gp_stats = pd.read_csv(PROJECT_DIR / "data" / "gp_summary_stats_seed0.csv")
robust_gps = set(gp_stats.loc[gp_stats["match_count"] >= 5, "gp_idx"])
filtered_df = best_corr_df[best_corr_df["best_rqvi_gp"].isin(robust_gps)].copy()
filtered_df = filtered_df.sort_values("best_corr").reset_index(drop=True)
print(f"Filtered to {len(filtered_df)} pairs with robust RQVI GPs (match_count >= 5)")
bottom10 = filtered_df.iloc[:10].copy()
print("Bottom 10 pairs:")
print(bottom10.to_string(index=False))

PAIRS = []
for _, row in bottom10.iterrows():
    PAIRS.append({
        "flashier_factor": row["flashier_factor"],     # e.g. "F51"
        "rqvi_gp": int(row["best_rqvi_gp"]),
        "best_corr": row["best_corr"],
    })

rqvi_gps = sorted(set(p["rqvi_gp"] for p in PAIRS))
flashier_factors = sorted(set(p["flashier_factor"] for p in PAIRS))
print(f"Unique RQVI GPs: {rqvi_gps}")
print(f"Unique Flashier factors: {flashier_factors}")

# ─── Step 1: metadata ────────────────────────────────────────────────────────
print("Loading metadata...")
obs = load_main_obs()

# ─── Step 2: Flashier cell loadings & cluster means ─────────────────────────
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

# ─── Step 5: Extract RQVI & Flashier loadings for all 10 pairs ──────────────
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

# ─── Step 7: Plot ────────────────────────────────────────────────────────────
print("Plotting...")
n_pairs = len(PAIRS)
fig = plt.figure(figsize=(16, 3 * n_pairs + 3))
gs = fig.add_gridspec(
    n_pairs + 1, 4,
    height_ratios=[3] * n_pairs + [2.5],
    hspace=0.35, wspace=0.30,
)

for row_i, p in enumerate(PAIRS):
    gp = p["rqvi_gp"]
    ff = p["flashier_factor"]
    # Factor number for flashier_gene column (e.g. "F51" -> "V51")
    ff_num = ff[1:]  # strip "F"
    label = f"GP {gp} vs {ff} (r={p['best_corr']:.3f})"
    print(f"  Row {row_i}: {label}")

    # Col 0: RQVI UMAP
    ax = fig.add_subplot(gs[row_i, 0])
    vals = rqvi_loadings[gp][shuffle_order]
    vmax = np.percentile(vals[vals > 0], 99) if (vals > 0).any() else 1.0
    ax.scatter(
        umap_coords[shuffle_order, 0], umap_coords[shuffle_order, 1],
        s=DOT_SIZE, c=vals, cmap="viridis", vmin=0, vmax=vmax,
        alpha=0.6, rasterized=True,
    )
    ax.set_title(f"RQVI GP {gp}", fontsize=8, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if row_i == 0:
        ax.set_ylabel("UMAP", fontsize=9)

    # Col 1: Flashier UMAP
    ax = fig.add_subplot(gs[row_i, 1])
    vals = flashier_loadings[ff][shuffle_order]
    vmax = np.percentile(vals[vals > 0], 99) if (vals > 0).any() else 1.0
    ax.scatter(
        umap_coords[shuffle_order, 0], umap_coords[shuffle_order, 1],
        s=DOT_SIZE, c=vals, cmap="viridis", vmin=0, vmax=vmax,
        alpha=0.6, rasterized=True,
    )
    ax.set_title(f"Flashier {ff}", fontsize=8, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Col 2: RQVI MD plot
    ax = fig.add_subplot(gs[row_i, 2])
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
            fontsize=5.5, color="#1f77b4", alpha=0.9,
            xytext=(3, 3), textcoords="offset points",
        )
    ax.set_title(f"RQVI GP {gp}", fontsize=8, fontweight="bold", loc="left")
    ax.set_xlabel("Gene effect", fontsize=7)
    if row_i == 0:
        ax.set_ylabel("Mean log expr", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Col 3: Flashier MD plot
    ax = fig.add_subplot(gs[row_i, 3])
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
            fontsize=5.5, color="#1f77b4", alpha=0.9,
            xytext=(3, 3), textcoords="offset points",
        )
    ax.set_title(f"Flashier {ff}", fontsize=8, fontweight="bold", loc="left")
    ax.set_xlabel("Gene effect", fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ── Last row: cluster-level similarity histogram ─────────────────────────────
ax_hist = fig.add_subplot(gs[n_pairs, :])
ax_hist.hist(best_corr_df["best_corr"], bins=30,
             color="#4C72B0", edgecolor="white", linewidth=0.5, alpha=0.7)

colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, p in enumerate(PAIRS):
    ax_hist.axvline(
        p["best_corr"], color=colors[i], linestyle="--", linewidth=1.2,
        label=f"GP {p['rqvi_gp']} vs {p['flashier_factor']} ({p['best_corr']:.2f})",
    )
ax_hist.set_xlabel("Best-match Pearson r (cluster-level)", fontsize=9)
ax_hist.set_ylabel("Count (Flashier factors)", fontsize=9)
ax_hist.set_title("Cluster-level similarity (n=200) — bottom 10 pairs (robust GPs only, match_count≥5)",
                  fontsize=10, fontweight="bold", loc="left")
ax_hist.legend(fontsize=5.5, loc="upper left", ncol=2)
ax_hist.set_xlim(-1, 1)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)

# ─── Save ────────────────────────────────────────────────────────────────────
outpath = FIG_DIR / "bottom10_pairs.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=200)
print(f"Saved figure to {outpath}")
plt.close(fig)
print("Done!")
