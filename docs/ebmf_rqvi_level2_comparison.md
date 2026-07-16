# EBMF–RQVI factor comparison across level2 clusters

## Overview

This figure compares the cluster-level activity patterns of 200 EBMF factors with 200 corresponding RQVI factors. The two heatmaps use the same row order and the same 114 level2-cluster columns, so matching biological patterns can be compared directly from left to right.

The visual design follows Figure 1C. Both matrices use a white-to-blue loading scale, broad lineage annotations are shown above the columns, and factor identifiers are omitted. The figure contains no title, seed labels, correlation bars, or per-row annotations. Only the y-axis labels, `EBMF factors` and `Corresponding RQVI factors`, are retained.

## Reproduction

The default command reads the two saved display matrices and redraws the figure without recalculating factor matches:

```bash
python scripts/fig_ebmf_rqvi_level2_comparison.py
```

Recompute the one-to-one matching and overwrite the saved display matrices only when the underlying data or matching method changes:

```bash
python scripts/export_ebmf_level2_cluster_means.py
python scripts/fig_ebmf_rqvi_level2_comparison.py --recompute-matches
```

## Figure files

- `figures/main_figures/ebmf_rqvi_level2_comparison.pdf`: manuscript-ready figure.
- `figures/main_figures/ebmf_rqvi_level2_comparison.png`: 300-dpi preview.
- `scripts/fig_ebmf_rqvi_level2_comparison.py`: complete plotting and matching code.

## Data files

- `data/ebmf_mean_loadings_by_level2_cluster.csv`: raw mean loadings for 114 clusters and 200 EBMF factors.
- `data/rqvi_multiseed_mean_loadings_by_level2_cluster.csv`: raw mean loadings for 114 clusters and all 2,560 seed-specific RQVI candidates.
- `data/ebmf_rqvi_multiseed_level2_one_to_one_matches.csv`: the 200 selected EBMF–RQVI pairs, their Pearson correlations, and the display order.
- `data/ebmf_level2_scaled_loadings_for_comparison.csv`: the exact 200 × 114 matrix displayed in the left heatmap.
- `data/matched_rqvi_multiseed_level2_scaled_loadings_for_comparison.csv`: the exact 200 × 114 matrix displayed in the right heatmap.
- `data/rqvi_multiseed_candidate_metadata.csv`: RQVI program identifiers and the nonzero-variance eligibility flag.

## How the figure is generated

The complete workflow is implemented in `scripts/fig_ebmf_rqvi_level2_comparison.py`:

```text
cell-level loadings
    → mean loading per level2 cluster
    → EBMF × RQVI correlation matrix
    → global one-to-one factor assignment
    → common row and cluster order
    → within-factor 0–1 scaling
    → saved display matrices
    → PDF and PNG
```

### 1. Calculate cluster-mean loadings

The EBMF cell-loading matrix is restricted to the 633,684 cells used in the RQVI analysis. Loadings are averaged within the 114 values of `Cluster_totalvi20240525rmigtsample_Res0.5`, producing `data/ebmf_mean_loadings_by_level2_cluster.csv` with shape 114 clusters × 200 EBMF factors.

The 114 × 256 cluster-mean RQVI matrices from 10 model runs are concatenated column-wise into `data/rqvi_multiseed_mean_loadings_by_level2_cluster.csv`, which has shape 114 × 2,560. The run and RQVI factor identifiers are retained in this source matrix for reproducibility, but they are not displayed in the figure.

### 2. Determine the 200 corresponding RQVI factors

Every EBMF and RQVI factor is independently z-scored across the 114 clusters. The signed Pearson correlation matrix is then calculated as `EBMF_z.T @ RQVI_z / 114`, giving a 200 × 2,560 correlation matrix. Six RQVI candidates have constant cluster profiles and are excluded because their Pearson correlations are undefined.

The remaining correlations are passed to `scipy.optimize.linear_sum_assignment`. The algorithm maximizes the sum of signed correlations while assigning each of the 200 EBMF factors to a different RQVI candidate. The selected pairs and their correlations are saved in `data/ebmf_rqvi_multiseed_level2_one_to_one_matches.csv`.

### 3. Set the row and column order

EBMF rows are ordered by average-linkage hierarchical clustering with correlation distance and optimal leaf ordering. The corresponding RQVI factor for each EBMF factor is placed on the same row in the right heatmap. Thus row `i` on the right is always the assigned counterpart of row `i` on the left.

Cluster columns follow `data/rqvi_seed0_level2_heatmap_cluster_order.csv`, sorted by `display_column`. Both heatmaps therefore have identical column positions. The colored strip above each heatmap comes from the `level1` annotation in this file. Thin vertical lines mark boundaries between level1 groups.

### 4. Create the matrices used directly for plotting

Matching is performed with z-scored values, but the visual style follows the loading heatmap in Figure 1C. After matching and ordering, each factor is independently min–max scaled across clusters:

```text
relative_loading = (loading - minimum_loading) / (maximum_loading - minimum_loading)
```

This produces values between 0 and 1 without changing Pearson correlation. The exact matrices passed to `imshow` are saved as:

- `data/ebmf_level2_scaled_loadings_for_comparison.csv`: 200 EBMF rows × 114 cluster columns.
- `data/matched_rqvi_multiseed_level2_scaled_loadings_for_comparison.csv`: 200 corresponding RQVI rows × the same 114 cluster columns.

The CSV row identifiers preserve factor correspondence for auditing, but the plotting code intentionally does not render them.

### 5. Draw the figure

The default plotting path reads only the two saved 200 × 114 display matrices and the cluster-order table. It does not recalculate correlations or factor matches. Both matrices are drawn with Matplotlib `imshow`, the sequential `Blues` colormap, `vmin=0`, `vmax=1`, automatic aspect ratio, and rasterized heatmap bodies. A shared horizontal colorbar reports `Relative loading`.

The figure contains no overall title, panel title, factor identifiers, cluster tick labels, correlation bars, or run/seed annotations. The only y-axis text is `EBMF factors` and `Corresponding RQVI factors`. Broad lineage labels are shown above the matrices to match the visual organization of Figure 1C; a lineage spanning fewer than four columns remains visible in the color strip but is not labeled to prevent overlap.

The canvas size is 10.8 × 7.8 inches. The script writes a one-page PDF with vector labels and rasterized heatmap bodies, plus a 300-dpi PNG preview.

## Summary

The median correlation among the 200 assigned pairs was 0.743, and 93.0% of pairs had `r ≥ 0.5`. The unconstrained pooled best-match median was 0.752, so enforcing one-to-one correspondence reduced the mean correlation by only 0.019. This indicates that the agreement is not driven primarily by repeatedly assigning many EBMF factors to the same RQVI candidate.

The comparison supports the conclusion that the RQVI ensemble recovers cluster-associated activity patterns similar to those identified by EBMF. Because the candidate pool combines multiple model runs, the result should be interpreted as ensemble-level recovery rather than recovery by every individual run or proof of 200 distinct biological processes.

## Draft caption

**Cluster-level activity profiles of corresponding EBMF and RQVI factors.** Cell loadings were averaged within 114 fine-grained T-cell clusters. The left heatmap shows 200 EBMF factors ordered by hierarchical clustering of their cluster-level profiles. The right heatmap shows the corresponding RQVI factors selected by a global one-to-one maximum-correlation assignment, using the same row and cluster order. Loadings were independently rescaled from 0 to 1 within each factor for visualization. Broad lineage annotations are shown above the columns. The median assigned Pearson correlation was 0.743, and 93.0% of pairs had `r ≥ 0.5`.
