# Best-Match Signed Pearson r Histogram

> **Main figure — best-match Pearson r distribution.** Quantitative companion to panel b (cell-level Flashier × RQVI similarity heatmap, TBD), summarising how well RQVI matches Flashier across all 200 Flashier factors. See the panel-level storyline in [`figures_summary.md`](figures_summary.md).

Multi-seed best-match signed Pearson r distribution across all 200 Flashier factors, using 10 RQVI seeds. A plain histogram — no shaded region, no vertical pair markers.

## Description

For each of the 200 Flashier factors, the script computes the cluster-level Z-scored Pearson r against every RQVI GP across all 10 seeds. The best (maximum positive) correlation per Flashier factor across all seeds is retained. The resulting distribution is plotted as a histogram spanning r from -1 to 1.

## Inputs

| Path | What |
|------|------|
| `/data/tianzew/immgenT/david_final_10k_genes.h5ad` | Main dataset (obs metadata) via `utils.py` |
| `.../Evaluation/Subcluster/cell_factor_matrix.txt` | Flashier cell loadings — used to compute cluster means |
| `.../Evaluation/function_analysis/corr_rst/rqvi_seed{0-9}_gp_cell_level.csv` | RQVI cluster-level mean loadings (10 seeds) |

## Script

`scripts/fig_hist_standalone.py` -> `figures/hist_standalone.pdf`

## How to reproduce

```bash
python scripts/fig_hist_standalone.py
```

No upstream dependencies — the script computes correlations inline.
