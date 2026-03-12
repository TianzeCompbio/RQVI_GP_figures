# Best-Match Signed Pearson r Histogram

Multi-seed best-match signed Pearson r distribution across all 200 Flashier factors, using 10 RQVI seeds.

## Description

For each of the 200 Flashier factors, the script computes the cluster-level Z-scored Pearson r against every RQVI GP across all 10 seeds. The best (maximum positive) correlation per Flashier factor across all seeds is retained. The resulting distribution is plotted as a histogram spanning r from -1 to 1.

Key features:
- Green shaded region marks r >= 0.5 ("covered" factors)
- Vertical dashed lines mark the two highlighted pairs: GP 38 vs F58 (r=0.573) and GP 45 vs F35 (r=0.436)
- Annotation shows the percentage of Flashier factors covered at the r >= 0.5 threshold

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
