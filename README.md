# RQVI vs Flashier: Gene Program Comparison Figures

Figures for the multi-panel comparison of **RQVI** (Randomized Quasi-Variational Inference) and **Flashier** gene program methods. RQVI learns sparse, interpretable gene programs from single-cell expression data via variational inference with randomized sparsity priors. This figure set evaluates RQVI against Flashier across scalability, sparsity, coverage, and per-program agreement.

## Repository structure

```
figures_version_v2/
├── docs/           # Per-figure documentation
├── scripts/        # Figure-generation scripts + utils.py
├── figures/        # Output PDFs and schematic PNGs
└── data/           # Intermediate CSVs and input data
```

## Figure overview

| Figure | Description | Script | Output | Doc |
|--------|-------------|--------|--------|-----|
| Method schematic | RQVI architecture & inference | — (manual) | `figures/RQVI_figure_method_new.pdf` | [`docs/method_schematic.md`](docs/method_schematic.md) |
| Scalability | Training time vs dataset size | `fig_scalability.py` | `figures/scalability.pdf` | [`docs/scalability.md`](docs/scalability.md) |
| Sparsity scatter | Cell sparsity vs gene sparsity | `fig_gp_sparsity_scatter.py` | `figures/gp_sparsity_scatter.pdf` | [`docs/sparsity_scatter.md`](docs/sparsity_scatter.md) |
| Correlation histogram | Best-match signed Pearson r distribution | `fig_hist_standalone.py` | `figures/hist_standalone.pdf` | [`docs/correlation_histogram.md`](docs/correlation_histogram.md) |
| Pair GP45 vs F35 | UMAP + MD comparison (r=0.436) | `fig_pair_GP45_F35.py` | `figures/pair_GP45_F35.pdf` | [`docs/pair_GP45_F35.md`](docs/pair_GP45_F35.md) |
| RQVI level2-cluster heatmap | Mean loading of all 256 RQVI GPs across 114 fine-grained clusters | `fig_rqvi_level2_cluster_heatmap.py` | `figures/main_figures/rqvi_level2_cluster_heatmap.pdf` | [`docs/rqvi_level2_cluster_heatmap.md`](docs/rqvi_level2_cluster_heatmap.md) |
| Aligned EBMF–RQVI cluster comparison | 200 EBMF factors and 200 distinct matches from 10 pooled RQVI seeds across the same 114 clusters | `fig_ebmf_rqvi_level2_comparison.py` | Combined PDF plus `figures/main_figures/ebmf_rqvi_level2_comparison_subfigures/` | [`docs/ebmf_rqvi_level2_comparison.md`](docs/ebmf_rqvi_level2_comparison.md) |
| Best-match 4 factors | Best-match RQVI GPs for 4 Flashier factors (UMAP + MD) | `fig_rqvi_best_match_4factors.py` | `figures/rqvi_best_match_4factors.png` (PNG; PDF too large at 6.7 MB) | — |
| Coverage | Flashier GP coverage vs RQVI seeds | `fig_rqvi_flashier_coverage.py` | `figures/rqvi_flashier_coverage.pdf` | [`docs/coverage.md`](docs/coverage.md) |

## How to reproduce

### Dependencies

- numpy, pandas, matplotlib, scanpy, scipy, anndata, h5py

The level2-cluster heatmap is self-contained: its losslessly compressed source matrix is included as `data/rqvi_cell_loadings_seed0.h5`, with exact cell names, program names/indices, and cluster annotations. The aligned EBMF–RQVI comparison can be redrawn from its two final 200 × 114 display matrices or recomputed from the bundled EBMF and pooled RQVI cluster-mean matrices and candidate metadata under `data/`.

### External data

Most scripts depend on large external files. The level2-cluster heatmap and both modes of the aligned comparison use data bundled in this repository. External inputs are needed only to regenerate those bundled cluster-mean inputs from cell-level data or to run the other analyses.

| Path | What | Used by |
|------|------|---------|
| `/data/tianzew/immgenT/david_final_10k_genes.h5ad` | Main dataset (obs metadata + expression) | All scripts via `utils.py` |
| `/data/tianzew/immgenT/RQVI_multiseeds/results/cmtloss08_64by4GPs_seed{0-9}.h5ad` | RQVI cell loadings (seed 0 via `utils.py`; all 10 seeds for multi-seed analyses) | Correlation histogram, pair GP45 vs F35, coverage |
| `/data/tianzew/immgenT/RQVI_multiseeds/results/cmtloss08_64by4GPs_seed0.h5ad` | Original source used only when rebuilding the bundled compressed exchange file | `export_rqvi_cell_loadings.py` |
| `/data/tianzew/immgenT/RQVI/cmtloss08_64by4GPs_mde_totalVI.h5ad` | UMAP coordinates for pair plots | Pair GP45 vs F35 |
| `/data/tianzew/immgenT/totalvi_20241006_mde.csv` | MDE coordinates | `utils.py` |
| `.../Evaluation/function_analysis/corr_rst/rqvi_seed{0-9}_gp_cell_level.csv` | Original RQVI cluster-level mean loadings (all 10 seeds) | Upstream regeneration of the bundled pooled matrix, sparsity scatter, correlation histogram, coverage |
| `.../Evaluation/Subcluster/cell_factor_matrix.txt` | Flashier cell loadings (~2.7 GB) | Correlation histogram, pair GP45 vs F35, coverage, `export_ebmf_level2_cluster_means.py` |
| `.../Evaluation/Subcluster/gene_factor_matrix.txt` | Flashier gene effects (~81 MB) | Pair GP45 vs F35 |

Full base path for `.../Evaluation/` entries: `/homes/gws/tianzew/projects/gene_program_model/Evaluation/`

### Execution order

All figure scripts are independent and can run in any order. Note that `fig_scalability.py` depends on `data/scalability_benchmark.csv`, which is generated by `benchmark_scalability.py` (requires GPU).

### Running

Each figure has its own script in `scripts/`. All scripts are independent:

```bash
python scripts/fig_scalability.py              # -> figures/scalability.pdf
python scripts/fig_gp_sparsity_scatter.py      # -> figures/gp_sparsity_scatter.pdf
python scripts/fig_hist_standalone.py           # -> figures/hist_standalone.pdf
python scripts/fig_pair_GP45_F35.py            # -> figures/pair_GP45_F35.pdf
python scripts/fig_rqvi_level2_cluster_heatmap.py # -> figures/main_figures/rqvi_level2_cluster_heatmap.pdf/.png + data CSVs
python scripts/fig_ebmf_rqvi_level2_comparison.py # -> combined figure + standalone subfigure PDFs
python scripts/fig_rqvi_best_match_4factors.py # -> figures/rqvi_best_match_4factors.pdf (use .png version; PDF is 6.7 MB)
python scripts/fig_rqvi_flashier_coverage.py   # -> figures/rqvi_flashier_coverage.pdf
```

Refer to each figure's doc (linked above) for details on data inputs and parameters.
