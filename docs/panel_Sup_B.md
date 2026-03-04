# Panel Sup B — RQVI Coverage of Flashier GPs

Line plot showing the fraction of Flashier GPs covered as RQVI seeds are added greedily, across 9 correlation thresholds.

## Axes & encoding

- **X-axis** — Number of RQVI seeds (1–10), selected via greedy set cover
- **Y-axis** — Coverage = fraction of 200 Flashier GPs with best-match |r| ≥ threshold
- **Color** — Viridis colormap, 9 lines for thresholds 0.1–0.9
- **Markers** — Circles at each integer seed count

## Methodology

1. Flashier cell loadings aggregated by `obs[CLUSTER_COL]` → 114 clusters × 200 factors
2. RQVI cluster means loaded from pre-computed CSVs (10 seeds × 256 GPs × 114 clusters)
3. Both matrices Z-scored across clusters; Pearson correlation via dot product → best absolute correlation per Flashier GP per seed
4. Greedy seed selection: at each step, pick the seed maximising coverage (standard greedy set cover). A Flashier GP is "covered" if its best-match |r| ≥ threshold for any selected seed

## Input data

| File | Description |
|------|-------------|
| `cell_factor_matrix.txt` | ~633k cells × 200 factors; Flashier cell loadings (external) |
| `rqvi_seed{0-9}_gp_cell_level.csv` | 114 clusters × 256 GPs; RQVI cluster means, 10 files (external) |
| `david_final_10k_genes.h5ad` | Main dataset obs metadata with cluster labels via `load_main_obs()` (external) |

## Script & output

- **Script:** `scripts/fig_rqvi_flashier_coverage.py`
- **Output figure:** `figures/rqvi_flashier_coverage.pdf`

## How to reproduce

```bash
uv run python figures_version_v2/scripts/fig_rqvi_flashier_coverage.py
```
