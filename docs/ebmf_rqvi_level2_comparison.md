# EBMF–RQVI factor comparison across level2 clusters

## Overview

This figure compares the cluster-level activity patterns of 200 EBMF factors with 200 corresponding RQVI factors. The two heatmaps use the same row order and the same 114 level2-cluster columns, so matching biological patterns can be compared directly from left to right.

The visual design follows Figure 1C. Both matrices use a white-to-blue loading scale, broad lineage annotations are shown above the columns, and factor identifiers are omitted. The figure contains no title, seed labels, correlation bars, or per-row annotations. Only the y-axis labels, `EBMF factors` and `Corresponding RQVI factors`, are retained.

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

## Matching and display

EBMF loadings were averaged over the same 633,684 cells used in the RQVI analysis. The 256 RQVI factors from each of 10 model runs were pooled into 2,560 seed-specific candidates. Six candidates had constant cluster profiles and were excluded because Pearson correlation was undefined.

Each factor profile was z-scored across the 114 clusters, and signed Pearson correlations were calculated between all 200 EBMF factors and 2,554 eligible RQVI candidates. A maximum-weight bipartite assignment selected 200 non-reused RQVI candidates while maximizing the total correlation with the 200 EBMF factors.

Rows were ordered by hierarchical clustering of the EBMF profiles. For display, each selected factor was independently rescaled to the interval 0–1 across clusters. This affine rescaling preserves Pearson correlation while producing a loading-matrix appearance consistent with Figure 1C. White indicates the lowest relative loading for a factor and dark blue indicates the highest.

## Summary

The median correlation among the 200 assigned pairs was 0.743, and 93.0% of pairs had `r ≥ 0.5`. The unconstrained pooled best-match median was 0.752, so enforcing one-to-one correspondence reduced the mean correlation by only 0.019. This indicates that the agreement is not driven primarily by repeatedly assigning many EBMF factors to the same RQVI candidate.

The comparison supports the conclusion that the RQVI ensemble recovers cluster-associated activity patterns similar to those identified by EBMF. Because the candidate pool combines multiple model runs, the result should be interpreted as ensemble-level recovery rather than recovery by every individual run or proof of 200 distinct biological processes.

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

## Draft caption

**Cluster-level activity profiles of corresponding EBMF and RQVI factors.** Cell loadings were averaged within 114 fine-grained T-cell clusters. The left heatmap shows 200 EBMF factors ordered by hierarchical clustering of their cluster-level profiles. The right heatmap shows the corresponding RQVI factors selected by a global one-to-one maximum-correlation assignment, using the same row and cluster order. Loadings were independently rescaled from 0 to 1 within each factor for visualization. Broad lineage annotations are shown above the columns. The median assigned Pearson correlation was 0.743, and 93.0% of pairs had `r ≥ 0.5`.
