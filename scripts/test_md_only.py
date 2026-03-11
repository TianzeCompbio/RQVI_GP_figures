"""
Quick MD-only preview: 2×2 gene-effect scatter (no UMAP, no cell loadings).

Top row: GP38 (RQVI) vs F58 (Flashier)
Bottom row: GP45 (RQVI) vs F35 (Flashier)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
import scanpy as sc
from utils import load_gene_effects, md_scatter, FIG_DIR, PATH_MAIN_H5AD

PATH_FLASHIER_GENE = "/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/gene_factor_matrix.txt"

# ─── Load gene effects ──────────────────────────────────────────────────────
print("Loading gene effects...")
rqvi_effects = load_gene_effects()
print(f"  RQVI gene effects: {rqvi_effects.shape}")

flashier_gene = pd.read_csv(PATH_FLASHIER_GENE, sep="\t", index_col=0)
print(f"  Flashier gene effects: {flashier_gene.shape}")

# ─── Compute mean log expression per gene (chunked backed mode) ─────────────
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

# ─── Intersect genes ────────────────────────────────────────────────────────
common_genes = (
    rqvi_effects.index
    .intersection(flashier_gene.index)
    .intersection(mean_log_expr.index)
)
print(f"  Common genes: {len(common_genes)}")
rqvi_effects = rqvi_effects.loc[common_genes]
flashier_gene = flashier_gene.loc[common_genes]
mean_expr = mean_log_expr.loc[common_genes]

# ─── Plot 2×2: GP38 top, GP45 bottom ────────────────────────────────────────
print("Plotting 2×2 MD-only panel...")
pairs = [
    (38, 58),  # top row
    (45, 35),  # bottom row
]

fig, axes = plt.subplots(2, 2, figsize=(16, 13))

for row, (rqvi_gp, flash_f) in enumerate(pairs):
    md_scatter(axes[row, 0], rqvi_effects[str(rqvi_gp)], mean_expr,
               f"RQVI GP {rqvi_gp}")
    axes[row, 0].set_ylabel("Mean log expr", fontsize=11)

    md_scatter(axes[row, 1], flashier_gene[f"V{flash_f}"], mean_expr,
               f"Flashier F{flash_f}")
    axes[row, 1].set_ylabel("")

fig.tight_layout(h_pad=2.0, w_pad=2.0)
outpath = FIG_DIR / "test_md_only.png"
fig.savefig(outpath, bbox_inches="tight", dpi=200)
print(f"Saved figure to {outpath}")
plt.close(fig)
print("Done!")
