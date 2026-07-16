# RQVI loading heatmap across level2 clusters

## Purpose

This figure is the RQVI counterpart to the EBMF loading heatmap in Figure 1c.
It shows whether RQVI independently recovers broad, lineage-associated, and
fine-cluster-associated activity patterns when the cell loadings are summarized
at the same biological resolution.

For the direct row-aligned comparison of EBMF factors and their best RQVI
matches, see [`ebmf_rqvi_level2_comparison.md`](ebmf_rqvi_level2_comparison.md).

## Output

- `figures/main_figures/rqvi_level2_cluster_heatmap.pdf`: editable figure for
  the manuscript layout.
- `figures/main_figures/rqvi_level2_cluster_heatmap.png`: 300-dpi preview.
- `data/rqvi_seed0_mean_loadings_by_level2_cluster.csv`: the exact 114 cluster
  x 256 GP arithmetic means plotted in the heatmap. Columns are explicitly
  named `GP0` through `GP255`.
- `data/rqvi_seed0_level2_heatmap_gp_order.csv`: displayed row order and the
  cluster with the maximum mean loading for each GP.
- `data/rqvi_seed0_level2_heatmap_cluster_order.csv`: displayed column order,
  parent level1 lineage, and number of cells contributing to each mean.

## Complete cell-level data

`data/rqvi_cell_loadings_seed0.h5` is a losslessly compressed exchange file
containing the complete seed-0 matrix used to calculate the cluster means. It
contains 633,684 cells x 256 programs and preserves the source `float32` values
exactly; no thresholding, rounding, or down-casting was applied.

```python
import h5py

with h5py.File("data/rqvi_cell_loadings_seed0.h5", "r") as f:
    loadings = f["cell_loadings"][:]              # (633684, 256), float32
    cell_names = f["metadata/cell_names"].asstr()[:]
    program_names = f["metadata/program_names"].asstr()[:]
    program_indices = f["metadata/program_indices"][:]

    cluster_codes = f["metadata/level2_cluster_codes"][:]
    cluster_labels = f["metadata/level2_cluster_labels"].asstr()[:]
    cell_clusters = cluster_labels[cluster_codes]
```

The file also includes `level1_codes`, `level1_labels`, and
`level2_cluster_level1_codes`, which reproduce the colored lineage annotation
above the heatmap.

## Method

1. Use seed 0 of RQVI (`cmtloss08_64by4GPs_seed0.h5ad`).
2. Treat `Cluster_totalvi20240525rmigtsample_Res0.5` as the fine-grained
   **level2 cluster** annotation (114 clusters). This is distinct from the
   four-valued `level2.group` field (`resting`, `activated`, `preT`, `nan`).
3. For every cluster and every GP, calculate the unweighted arithmetic mean of
   the cell-level loading across all cells in that cluster. All 633,684 cells
   contribute to exactly one cluster mean.
4. Group heatmap columns by parent level1 lineage, then naturally sort cluster
   numbers within each lineage. Program rows are ordered by average-linkage
   hierarchical clustering with cosine distance applied to L2-normalized
   cluster-mean profiles. This normalization controls **row ordering only**.
5. Display the raw, untransformed mean loadings with a sequential blue scale.
   The color scale is capped at the 99.5th percentile of positive values
   (`0.260551`) so a few large means do not hide lower-loading patterns. Values
   above the cap saturate in the plot; every uncapped value remains available
   in the CSV.

The generated means agree with the earlier independently generated file
`rqvi_seed0_gp_cell_level.csv` to a maximum absolute difference of
`5.8e-08` (float serialization precision).

## Reproduction

The bundled HDF5 archive makes the plot self-contained:

```bash
python scripts/fig_rqvi_level2_cluster_heatmap.py
```

To rebuild the exchange file from the original RQVI AnnData object:

```bash
python scripts/export_rqvi_cell_loadings.py --overwrite
python scripts/fig_rqvi_level2_cluster_heatmap.py
```

The exporter expects the original seed-0 AnnData file at
`/data/tianzew/immgenT/RQVI_multiseeds/results/cmtloss08_64by4GPs_seed0.h5ad`.
Both scripts accept command-line arguments to override input and output paths.

## Draft caption

**Mean RQVI gene-program loading across level2 clusters.** Heatmap showing the
mean cell loading of each of 256 RQVI gene programs (rows) in each of 114
fine-grained T-cell clusters (columns). Clusters are grouped by parent lineage,
shown by the colored annotation bar, and RQVI programs are hierarchically
ordered by their cluster-loading profiles. Color represents the untransformed
arithmetic mean loading; the display scale is capped at its 99.5th percentile.
