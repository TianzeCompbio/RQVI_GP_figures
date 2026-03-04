# Panel D–E — Cross-Method Similarity (RQVI vs Flashier) (for reference not final version)

Histogram of best-match Pearson correlations between 200 Flashier factors and 256 RQVI GPs, plus a UMAP grid showing the 10 most different pairs.

## Axes & encoding

### Histogram (top panel)

- **X-axis** — Best-match Pearson correlation (cluster-aggregated) for each Flashier factor
- **Y-axis** — Count of Flashier factors
- **Dashed line** — Median correlation across all 200 factors
- **Red rug ticks** — Bottom 10 factors (lowest correlation with any RQVI GP)

### UMAP grid (bottom panels)

- **Row 1** — RQVI GP cell loadings (best-matching GP for each bottom-10 Flashier factor)
- **Row 2** — Flashier factor cell loadings
- **Color** — Viridis, scaled per panel (0 to 99th percentile of positive values)
- **Columns** — Sorted by correlation (ascending, left = most different)
- **Red label** — Pearson r between the pair (cluster-aggregated)

## Correlation methodology

### Primary: cluster-level (114 clusters)

1. RQVI cluster means loaded directly from pre-computed CSV (114 clusters × 256 GPs)
2. Flashier cell loadings aggregated by `obs[CLUSTER_COL]` → 114 clusters × 200 factors
3. Both matrices Z-scored across clusters, then dot-product → 256 × 200 correlation matrix
4. Best RQVI GP selected per Flashier factor (max absolute correlation)

### Secondary: cell-level

1. RQVI cell loadings from h5ad and Flashier cell loadings intersected to common cells (~633k)
2. Z-scored across cells, dot-product → 256 × 200 correlation matrix
3. Saved to CSV for reference (not used in figure)

## Input data

| File | Description |
|------|-------------|
| `rqvi_seed0_gp_cell_level.csv` | 114 clusters × 256 GPs; RQVI cluster means (external) |
| `cell_factor_matrix.txt` | ~633k cells × 200 factors; Flashier cell loadings (external) |
| `david_final_10k_genes.h5ad` | Main dataset obs metadata with cluster labels (external) |
| `cmtloss08_64by4GPs_seed0.h5ad` | RQVI cell loadings for UMAP (external) |
| `totalvi_20241006_mde.csv` | UMAP/MDE coordinates (external) |

## Script & output

- **Script:** `scripts/fig_cross_method_similarity.py`
- **Intermediate CSVs:**
  - `data/cross_method_best_corr.csv` — 200 rows: `flashier_factor`, `best_rqvi_gp`, `best_corr` (cluster-level)
  - `data/cross_method_best_corr_cell_level.csv` — 200 rows: same columns (cell-level)
- **Output figure:** `figures/cross_method_similarity_comprehensive_for_reference.pdf`

## How to reproduce

```bash
uv run python figures_version_v2/scripts/fig_cross_method_similarity.py
```
