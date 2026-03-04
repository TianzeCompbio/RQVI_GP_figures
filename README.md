# RQVI vs Flashier: Gene Program Comparison Figures

Figures for the multi-panel comparison of **RQVI** (Randomized Quasi-Variational Inference) and **Flashier** gene program methods. RQVI learns sparse, interpretable gene programs from single-cell expression data via variational inference with randomized sparsity priors. This figure set evaluates RQVI against Flashier across scalability, sparsity, coverage, and per-program agreement.

## Repository structure

```
figures_version_v2/
├── docs/           # Per-panel documentation
├── scripts/        # Figure-generation scripts + utils.py
├── figures/        # Output PDFs and schematic PNGs
└── data/           # Intermediate CSVs and input data
```

## Panel overview

| Panel | Description | Script | Doc | Status |
|-------|-------------|--------|-----|--------|
| A | RQVI architecture & inference schematics | — (manual) | [`docs/panel_A.md`](docs/panel_A.md) | Draft |
| B | Time scalability plot | — | — | Running |
| C | GP sparsity scatter | `fig_gp_sparsity_scatter.py` | [`docs/panel_C.md`](docs/panel_C.md) | Done |
| D-F | RQVI vs Flashier pair comparison (combined) | `fig_panel_D_F.py` | [`docs/panel_D_F.md`](docs/panel_D_F.md) | Done |
| D | Similarity histograms (split) | `fig_panel_D_hist.py` | [`docs/panel_D_F.md`](docs/panel_D_F.md) | Done |
| E | GP38 vs F58: UMAP + MD (split) | `fig_panel_E_pair1.py` | [`docs/panel_D_F.md`](docs/panel_D_F.md) | Done |
| F | GP45 vs F35: UMAP + MD (split) | `fig_panel_F_pair2.py` | [`docs/panel_D_F.md`](docs/panel_D_F.md) | Done |
| Sup B | Flashier coverage (cluster-level) | `fig_rqvi_flashier_coverage.py` | [`docs/panel_Sup_B.md`](docs/panel_Sup_B.md) | Done |
| Sup B v2 | Flashier coverage (cell-level) | `fig_rqvi_flashier_coverage_cell_level.py` | [`docs/panel_Sup_B_v2.md`](docs/panel_Sup_B_v2.md) | Done |
| (ref) | Cross-method similarity + intermediate CSVs | `fig_cross_method_similarity.py` | — | Done |

Additional reference material: [`docs/for_reference_panel_D_to_F.md`](docs/for_reference_panel_D_to_F.md)

## How to reproduce

### Dependencies

- numpy, pandas, matplotlib, scanpy, scipy

### External data

Every script (except `fig_panel_D_hist.py`, which reads only local CSVs) depends on large external files. Verify access before running.

| Path | What | Used by |
|------|------|---------|
| `/data/tianzew/immgenT/david_final_10k_genes.h5ad` | Main dataset (obs metadata + expression) | `utils.py` → most scripts |
| `/data/tianzew/immgenT/RQVI_multiseeds/results/cmtloss08_64by4GPs_seed{0-9}.h5ad` | RQVI cell loadings (seed 0 via `utils.py`; all 10 seeds for coverage scripts) | `fig_cross_method_similarity.py`, `fig_panel_D_F.py`, `fig_panel_E_pair1.py`, `fig_panel_F_pair2.py`, `fig_rqvi_flashier_coverage_cell_level.py` |
| `/data/tianzew/immgenT/totalvi_20241006_mde.csv` | UMAP/MDE coordinates | `utils.py` → UMAP-based scripts |
| `.../Evaluation/function_analysis/corr_rst/rqvi_seed{0-9}_gp_cell_level.csv` | RQVI cluster-level mean loadings (seed 0 via `utils.py`; all 10 seeds for cluster coverage) | `fig_cross_method_similarity.py`, `fig_gp_sparsity_scatter.py`, `fig_panel_D_F.py`, `fig_panel_E_pair1.py`, `fig_panel_F_pair2.py`, `fig_rqvi_flashier_coverage.py` |
| `.../Evaluation/Subcluster/cell_factor_matrix.txt` | Flashier cell loadings (~2.7 GB) | `fig_cross_method_similarity.py`, `fig_panel_D_F.py`, `fig_panel_E_pair1.py`, `fig_panel_F_pair2.py`, `fig_rqvi_flashier_coverage.py`, `fig_rqvi_flashier_coverage_cell_level.py` |
| `.../Evaluation/Subcluster/gene_factor_matrix.txt` | Flashier gene effects (~81 MB) | `fig_panel_D_F.py`, `fig_panel_E_pair1.py`, `fig_panel_F_pair2.py` |

Full base path for `.../Evaluation/` entries: `/homes/gws/tianzew/projects/gene_program_model/Evaluation/`

### Execution order

Some scripts produce intermediate CSVs that downstream scripts depend on. Run them in this order:

```
fig_cross_method_similarity.py  →  data/cross_method_best_corr.csv
                                   data/cross_method_best_corr_cell_level.csv
        ↓
fig_panel_D_F.py                →  data/panel_D_F_pair_correlations.csv
        ↓
fig_panel_D_hist.py                (reads all 3 CSVs above)
fig_panel_E_pair1.py               (independent — can run in any order)
fig_panel_F_pair2.py               (independent — can run in any order)
```

All other scripts (`fig_gp_sparsity_scatter.py`, `fig_rqvi_flashier_coverage.py`, `fig_rqvi_flashier_coverage_cell_level.py`) are independent and can run in any order.

### Running

Each panel has its own script in `scripts/`. For example:

```bash
# Step 1: Generate cross-method correlation CSVs (required by panels D-F)
python scripts/fig_cross_method_similarity.py

# Step 2: Panels D-F combined (generates pair correlation CSV)
python scripts/fig_panel_D_F.py

# Step 3: Panels D-F individual split panels (depend on CSVs from steps 1-2)
python scripts/fig_panel_D_hist.py   # -> figures/panel_D_hist.pdf
python scripts/fig_panel_E_pair1.py  # -> figures/panel_E_pair1.pdf
python scripts/fig_panel_F_pair2.py  # -> figures/panel_F_pair2.pdf

# Independent scripts (no ordering constraints)
python scripts/fig_gp_sparsity_scatter.py
python scripts/fig_rqvi_flashier_coverage.py
python scripts/fig_rqvi_flashier_coverage_cell_level.py
```

Refer to each panel's doc (linked above) for details on data inputs and parameters.

## Open items / TODOs

1. **Panel A detail level** - Current schematics (`RQVI_architecture.png`, `RQVI_training.png`) may be too detailed. Consider whether a more simplified/brief version is needed.

2. **Panel B** - Time scalability plot is still running; not yet available.

3. **Panel D-F pair selection** - Current pairs (GP 38 <-> F58, GP 45 <-> F35) have moderate correlations (cluster: 0.57 / 0.44, cell: 0.35 / 0.15) rather than sitting at the bottom of the distribution. Need to decide whether to keep these pairs or select pairs with lower correlation to better illustrate differences between the methods.
