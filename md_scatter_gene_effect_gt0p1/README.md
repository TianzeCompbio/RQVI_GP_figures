# MD Scatter Gene-Effect Export (>0.1)

This folder contains exploratory versions of the Figure 7 / Figure S7 MD scatter panels with all genes above the requested effect threshold highlighted.

Threshold definition: genes are highlighted in blue when their positive gene effect/loading is greater than `0.1`. The labels are kept compact, following the style of the original MD scatter panels: only the top positive-effect genes plus manually highlighted genes are labelled. Grey points are all genes used in the MD scatter. The y-axis is mean log expression computed from the main immgenT h5ad file.

## Panel Summary

| panel | program | source_column | genes_gt_0.1 | max_gene_effect | source_path |
| --- | --- | --- | --- | --- | --- |
| Fig7f | RQVI GP45 (seed 0) | 45 | 155 | 0.746 | /homes/gws/tianzew/projects/GP_figures/data/RQVI_gene_factors/gp_effects_matrix_seed0.csv |
| Fig7g | Flashier F35 | V35 | 57 | 1.000 | /homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/gene_factor_matrix.txt |
| FigS7g | RQVI GP96 (seed 6), best match for Flashier F22 | 96 | 48 | 0.545 | /homes/gws/tianzew/projects/GP_figures/data/RQVI_gene_factors/gp_effects_matrix_seed6.csv |
| FigS7h | RQVI GP112 (seed 5), best match for Flashier F30 | 112 | 91 | 0.720 | /homes/gws/tianzew/projects/GP_figures/data/RQVI_gene_factors/gp_effects_matrix_seed5.csv |
| FigS7i | RQVI GP110 (seed 1), best match for Flashier F58 | 110 | 89 | 0.463 | /homes/gws/tianzew/projects/GP_figures/data/RQVI_gene_factors/gp_effects_matrix_seed1.csv |
| FigS7j | RQVI GP76 (seed 3), best match for Flashier F68 | 76 | 119 | 0.436 | /homes/gws/tianzew/projects/GP_figures/data/RQVI_gene_factors/gp_effects_matrix_seed3.csv |

## Files

- `plots/*.pdf` and `plots/*.png`: one original-style MD scatter per panel, with all >`0.1` genes highlighted.
- `tables/all_gene_values_for_md_scatter.csv`: long-format table with every gene used in every MD scatter.
- `tables/genes_effect_gt0p1.csv`: subset of genes with positive gene effect/loading > `0.1`.
- `tables/<panel>_*_gt0p1.csv`: per-panel threshold tables.
- `tables/mean_log_expression_used.csv`: cached mean log expression used on the y-axis.
- `tables/manual_highlight_gene_positions.csv`: lookup table for manually highlighted genes (`Cd8a, Cd8b1, Cd8b`) even when they do not pass the >`0.1` threshold.

## Manual Highlight Genes

To force specific genes to appear on the plots, edit the `MANUAL_HIGHLIGHT_GENES` list near the top of `scripts/export_md_threshold_gene_plots.py`.

For example:

```python
MANUAL_HIGHLIGHT_GENES = ["Cd8a", "Cd8b1", "Cd4", "Nkg7"]
```

Then rerun:

```bash
MPLCONFIGDIR=/tmp/mplconfig python scripts/export_md_threshold_gene_plots.py
```

Manual-highlight genes are labelled and circled in every panel where they are present. Genes with `gene_effect > 0.1` are still highlighted in blue; manual-highlight genes below the threshold are shown as white circles with dark outlines. Their exact coordinates are written to `tables/manual_highlight_gene_positions.csv`.

## Source Data

- Main expression h5ad for mean log expression: `/data/tianzew/immgenT/david_final_10k_genes.h5ad`
- Flashier gene effects: `/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/gene_factor_matrix.txt`
- RQVI gene effects: `/homes/gws/tianzew/projects/GP_figures/data/RQVI_gene_factors/gp_effects_matrix_seed{seed}.csv`

## Reproduction

Run from the `figures_version_v2` repository root:

```bash
MPLCONFIGDIR=/tmp/mplconfig python scripts/export_md_threshold_gene_plots.py
```
