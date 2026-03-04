# Panel D–F — RQVI vs Flashier Factor Pair Comparison

Side-by-side comparison of 2 user-selected factor pairs between RQVI (seed 0) and Flashier, with UMAP cell-loading maps, MD (mean–difference) gene-effect plots, and dual-level similarity histograms.

## Pairs

| Pair | RQVI GP | Flashier Factor |
|------|---------|-----------------|
| 1    | GP 38   | F58             |
| 2    | GP 45   | F35             |

## Axes & encoding

### UMAP panels (row 0, 4 panels)

- **Layout** — 2 pairs × 2 panels (RQVI GP, Flashier factor)
- **Color** — Viridis, scaled per panel from 0 to the 99th percentile of positive cell-loading values
- **Points** — ~100k cells (stratified downsample by level1), s=0.3, alpha=0.6, rasterized

### MD plots (row 1, 4 panels)

- **X-axis** — Gene effect weight (GP effect for RQVI; factor loading for Flashier)
- **Y-axis** — Mean log expression across all cells
- **Grey cloud** — All ~10k common genes
- **Blue circles** — Top 15 genes by absolute weight
- **Labels** — Top 8 genes by absolute weight

### Histograms (row 2, 2 panels)

- **Left** — Cluster-level best-match Pearson correlations (200 Flashier factors)
- **Right** — Cell-level best-match Pearson correlations (200 Flashier factors)
- **Blue bars** — Distribution of best-match correlations (30 bins)
- **Dashed vertical lines** — Pair-specific correlations (red = pair 1, blue = pair 2)

## Correlation methodology

### Cluster-level (114 clusters)

1. RQVI cluster means loaded from pre-computed CSV (114 clusters × 256 GPs)
2. Flashier cell loadings aggregated by cluster → 114 clusters × 200 factors
3. Both matrices Z-scored across clusters, dot-product → 256 × 200 correlation matrix
4. Specific pair correlations extracted from the full matrix

### Cell-level (2 pairs only)

1. RQVI and Flashier cell loadings intersected to common cells (~633k)
2. Per-pair: Z-score both vectors across cells, compute Pearson r via dot-product
3. Memory-efficient: only 2 correlations computed, not the full 256 × 200 matrix

## Input data

| File | Description |
|------|-------------|
| `rqvi_seed0_gp_cell_level.csv` | 114 clusters × 256 GPs; RQVI cluster means (external) |
| `cell_factor_matrix.txt` | ~633k cells × 200 factors; Flashier cell loadings (external) |
| `gene_factor_matrix.txt` | ~19.8k genes × 200 factors; Flashier gene effects (external) |
| `david_final_10k_genes.h5ad` | Main dataset; obs metadata + gene expression (external) |
| `cmtloss08_64by4GPs_seed0.h5ad` | RQVI cell loadings (external) |
| `totalvi_20241006_mde.csv` | UMAP/MDE coordinates (external) |
| `data/gp_effects_matrix_seed0.csv` | 19,805 genes × 256 GPs; RQVI gene effects |
| `data/cross_method_best_corr.csv` | 200 rows; cluster-level best-match correlations |
| `data/cross_method_best_corr_cell_level.csv` | 200 rows; cell-level best-match correlations |

## Script & output

- **Script:** `scripts/fig_panel_D_F.py`
- **Intermediate CSV:** `data/panel_D_F_pair_correlations.csv` — 2 rows: `rqvi_gp`, `flashier_factor`, `corr_cluster`, `corr_cell`
- **Output figure:** `figures/panel_D_F.pdf`

## How to reproduce

```bash
uv run python figures_version_v2/scripts/fig_panel_D_F.py
```
