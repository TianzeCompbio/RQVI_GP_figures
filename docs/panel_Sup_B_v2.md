# Panel Sup B v2 — RQVI Coverage of Flashier GPs (Cell-Level)

Line plot showing the fraction of Flashier GPs covered as RQVI seeds are added greedily, across 9 correlation thresholds. Correlations are computed at the **cell level** (no cluster aggregation).

## Axes & encoding

- **X-axis** — Number of RQVI seeds (1–10), selected via greedy set cover
- **Y-axis** — Coverage = fraction of 200 Flashier GPs with best-match |r| ≥ threshold
- **Color** — Viridis colormap, 9 lines for thresholds 0.1–0.9
- **Markers** — Circles at each integer seed count

## Methodology

1. Flashier cell loadings loaded from `cell_factor_matrix.txt` — kept as raw cell × 200 matrix (no aggregation)
2. RQVI cell loadings loaded from h5ad files (10 seeds × 256 GPs), one seed at a time to limit memory
3. Cells intersected across Flashier, all 10 RQVI h5ads, and main obs metadata
4. Both matrices Z-scored across cells; Pearson correlation via dot product → best absolute correlation per Flashier GP per seed
5. Greedy seed selection: at each step, pick the seed maximising coverage (standard greedy set cover). A Flashier GP is "covered" if its best-match |r| ≥ threshold for any selected seed

## Input data

| File | Description |
|------|-------------|
| `cell_factor_matrix.txt` | ~633k cells × 200 factors; Flashier cell loadings (external) |
| `cmtloss08_64by4GPs_seed{0-9}.h5ad` | ~633k cells × 256 GPs; RQVI cell loadings, 10 files (external) |
| `david_final_10k_genes.h5ad` | Main dataset obs metadata for cell intersection via `load_main_obs()` (external) |

## Script & output

- **Script:** `scripts/fig_rqvi_flashier_coverage_cell_level.py`
- **Output figure:** `figures/rqvi_flashier_coverage_cell_level.pdf`

## How to reproduce

```bash
uv run python figures_version_v2/scripts/fig_rqvi_flashier_coverage_cell_level.py
```
