# Best-Matched RQVI GPs for Four Flashier Factors

> **Sup figure panel c.** RQVI's best-of-10-seeds matches for **the four lineage-specific gene programs from Flashier's published figure 2** (F22, F30, F58, F68). The panel demonstrates that RQVI recovers each of Flashier's four canonical lineage GPs at high cross-method correlation (Pearson r ≈ 0.81–0.90 after multi-seed search). See the panel-level storyline in [`figures_summary.md`](figures_summary.md).

## Layout

A 2 × 4 figure (one column per Flashier factor):

- **Top row:** UMAP cell-loading plots of the best-match RQVI GP (purple colormap). Title shows `GP {gp} (seed {seed})` with a column header `F{factor} match (r=…)`.
- **Bottom row:** MD scatter (mean log expression vs gene effect) for the same RQVI GP, with the top-50 genes by absolute weight labelled.

The Flashier side is *not* drawn. The point of the panel is to document the RQVI program that Flashier's factor maps to.

## Best-match table (current figure)

Read directly from the figure title bars:

| Flashier | Best RQVI GP | Seed | Pearson r |
| --- | --- | --- | --- |
| F22 | GP 96 | 6 | 0.897 |
| F30 | GP 112 | 5 | 0.812 |
| F58 | GP 110 | 1 | 0.840 |
| F68 | GP 76 | 3 | 0.840 |

For comparison, the seed-0-only matches in `data/cross_method_best_corr.csv` are systematically lower:

| Flashier | Best RQVI GP (seed 0) | Pearson r (seed 0) |
| --- | --- | --- |
| F22 | GP 16 | 0.703 |
| F30 | GP 120 | 0.578 |
| F58 | GP 166 | 0.729 |
| F68 | GP 116 | 0.828 |

The gap between the seed-0 numbers and the best-of-10-seeds numbers is the load-bearing fact for the multi-seed-robustness story: a single RQVI run leaves real biology on the table; the multi-seed search closes the gap. This is why the panel exists alongside the scalability and coverage panels — they share the same speed → multi-seed → robustness chain.

## Methodology

1. Load Flashier cell loadings (~633k cells × 200 factors); aggregate to cluster means using `obs[CLUSTER_COL]`.
2. Load all 10 RQVI cluster-mean CSVs (one per seed).
3. For each target Flashier factor, Z-score the cluster-mean profile across clusters; for each RQVI seed, Z-score the RQVI cluster-mean matrix column-wise; compute Pearson r as the dot product divided by the number of clusters.
4. Take `argmax` across all 256 GPs × 10 seeds → record `(seed, gp, r)`.
5. Load the chosen RQVI cell loadings and gene effects (one h5ad per unique seed) and plot UMAP + MD.

The metric is positive Pearson r (not `|r|`); the script uses `np.argmax(corrs)`, not `np.argmax(np.abs(corrs))`.

## Why these four Flashier factors?

These are **the four lineage-specific gene programs from Flashier's published figure 2** — i.e. the canonical Flashier-discovered lineage GPs. The panel pivots on Flashier and asks: "for each of Flashier's four lineage-defining GPs, does RQVI find a comparable program?". The answer is yes, with Pearson r in the 0.81–0.90 range after multi-seed search.

The lead genes for each of F22, F30, F58, F68 can be read from `gene_factor_matrix.txt` (~81 MB external file).

## Inputs

| Path | What |
| --- | --- |
| `/data/tianzew/immgenT/david_final_10k_genes.h5ad` | Main dataset (obs metadata for cluster labels) via `utils.py` |
| `/homes/gws/tianzew/projects/gene_program_model/Evaluation/Subcluster/cell_factor_matrix.txt` | Flashier cell loadings (~633k cells × 200 factors, ~2.7 GB) |
| `/homes/gws/tianzew/projects/gene_program_model/Evaluation/function_analysis/corr_rst/rqvi_seed{0-9}_gp_cell_level.csv` | Pre-computed RQVI cluster means for all 10 seeds |
| `/data/tianzew/immgenT/RQVI/cmtloss08_64by4GPs_mde_totalVI.h5ad` | UMAP coordinates |
| `/data/tianzew/immgenT/RQVI_multiseeds/results/cmtloss08_64by4GPs_seed{0-9}.h5ad` | RQVI cell loadings (one per seed) |
| `data/RQVI_gene_factors/gp_effects_matrix_seed{0-9}.csv` | RQVI gene effects (one per seed) |

## Script & output

- **Script:** `scripts/fig_rqvi_best_match_4factors.py`
- **Output figure:** `figures/sup_figures/rqvi_best_match_4factors.pdf` (~6.7 MB) and `figures/sup_figures/rqvi_best_match_4factors.png` (preferred for embedding)
- **Hard-coded targets:** `TARGET_FACTORS = [22, 30, 58, 68]`, `N_SEEDS = 10`, `TARGET_N = 100_000` (stratified downsample for UMAP)

## How to reproduce

```bash
python figures_version_v2/scripts/fig_rqvi_best_match_4factors.py
```

The script's stdout prints the best-match table at the start of step 1; that print statement is the source of truth for the (seed, GP, r) values shown in the figure.

## Caveats

- The metric used here is *positive* Pearson r, but the coverage panel ([`coverage.md`](coverage.md)) uses `|r|` — the two panels are not strictly on the same metric.
- The panel does not show the Flashier side, so a reader cannot independently verify the match from the figure alone — they have to trust the r value in the column header.
- The PDF is 6.7 MB; use the PNG version for slide decks and submission unless vector quality is required.
