# RQVI Coverage of Flashier GPs

> **Sup figure panel d.** Cumulative coverage curve answering "how many RQVI seeds do we need to capture the Flashier factorization?". See the panel-level storyline in [`figures_summary.md`](figures_summary.md).

Line plot showing the fraction of Flashier GPs covered as RQVI seeds are added greedily, across four correlation thresholds. Because seeds are added one at a time via greedy set-cover and a Flashier factor stays covered the moment any selected seed clears the threshold, the coverage curve is **monotone non-decreasing** — i.e. it is a **cumulative coverage** plot.

## Axes & encoding

- **X-axis** — Number of RQVI seeds (1–10), selected via greedy set cover
- **Y-axis** — Coverage = fraction of 200 Flashier GPs with best-match |r| ≥ threshold
- **Color** — Blues colormap, 4 lines for thresholds in `THRESHOLDS = [0.3, 0.4, 0.5, 0.6]` (live in `scripts/fig_rqvi_flashier_coverage.py`)
- **Markers** — Circles at each integer seed count

> **Note (doc/script reconciliation):** an earlier version of this doc said 9 lines for thresholds 0.1–0.9. The live script uses 4 thresholds (0.3, 0.4, 0.5, 0.6); this doc has been updated to match. If the original 9-threshold intent should be restored, edit `THRESHOLDS` in the script accordingly.

> **Caveat:** the metric is `|r|` (signed magnitude), so anti-correlated programs count as covered. This is fine for an EBMF-style model where signs can flip across seeds, but worth flagging in the figure caption so a reader does not misread the number as a strict positive correlation.

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
