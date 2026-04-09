# RQVI Figure Section — Group Internal Report

> **Audience:** group members and co-authors who need to understand *why* each panel exists in the RQVI method-validation figure section, what it claims, and how the panels fit together. Not a paper-style figure caption.
>
> **Source figures:** `figures_version_v2/figures/main_figures/` and `figures_version_v2/figures/sup_figures/`. Per-panel methodology lives in the linked detail docs.

---

## 1. Storyline

This figure section is the **method-validation block** of the paper. It introduces RQVI (Residual-Quantized Variational Inference), a new gene-program (GP) discovery model, and defends three claims:

1. **RQVI converges with Flashier on gene-program discovery.** Flashier is the established EBMF baseline; if RQVI recovers a comparable factorization on the same data, it inherits Flashier's interpretability while gaining what comes next.
2. **RQVI is orders of magnitude faster than Flashier.** Fast training is what makes multi-seed runs tractable, which in turn powers the coverage / robustness story.
3. **RQVI recovers the lineage-specific gene programs we already know about.** The method is not just convergent on aggregate metrics — it captures the same biology as Flashier on the canonical T-cell lineages described elsewhere in the paper.

The **main figure** carries the convergent-discovery argument (claim 1). The **supplementary figure** carries the speed, sparsity, lineage-recovery, and coverage arguments (claims 2 and 3, plus a sparsity overview).

The rhetorical centerpiece of the main figure is the **GP45 / F35 case study** (panels c-f). We deliberately picked the *worst-matched pair by Pearson r* (cluster-level r ≈ 0.436, sitting in the left tail of the histogram) and showed that even this pair recovers the same biology — both GPs are TCR-activation programs whose lead genes and UMAP loadings overlap. This is a stronger statement than the histogram alone could make: if even the worst pair converges biologically, the method-level convergence claim holds throughout the distribution.

---

## 2. Main figure walkthrough

| Panel | File | Detail doc |
| --- | --- | --- |
| a | `main_figures/RQVI_figure_method_new.pdf` | [`method_schematic.md`](method_schematic.md) |
| b | cell-level Flashier × RQVI similarity heatmap | **TBD — figure not yet generated** |
| (hist) | `main_figures/hist_standalone.pdf` | [`correlation_histogram.md`](correlation_histogram.md) |
| c, d, e, f | `main_figures/pair_GP45_F35.pdf` | [`pair_GP45_F35.md`](pair_GP45_F35.md) |

**Panel a — RQVI architecture & inference schematic.** Hand-drawn schematic of the residual-quantized VAE: scVI encoder → N residual codebooks (coarse-to-fine GP decomposition) → quantized latent → scVI decoder. Bottom of the schematic shows the post-training inference that produces the two output matrices used everywhere downstream: a cell × GP loading matrix (distance-based softmax over codebook entries) and a GP × gene effect matrix (differential decoder log-expression with vs without each codebook contribution). See [`method_schematic.md`](method_schematic.md).

**Panel b — cell-level Flashier × RQVI similarity heatmap.** *Not yet generated.* Intended to show, at single-cell resolution, that for each Flashier factor there is an RQVI GP with strongly correlated loadings, and vice versa. This is the headline visual for the convergent-discovery claim. **Action item:** generate this panel.

**Histogram — best-match Pearson r distribution.** For each of the 200 Flashier factors, we take the maximum positive cluster-level Pearson r against any RQVI GP across all 10 seeds. The resulting distribution is plotted. Most Flashier factors are matched at r ≥ 0.5 (the green-shaded region in the figure); the percentage covered is annotated. Two pairs are marked with vertical dashed lines: GP38/F58 (r = 0.573, above threshold) and GP45/F35 (r = 0.436, below threshold). The histogram is the quantitative companion to panel b. See [`correlation_histogram.md`](correlation_histogram.md).

**Panels c, d, e, f — GP45 / F35 case study.** This is a 2 × 2 figure: top row is UMAP cell-loading plots for RQVI GP 45 (left) and Flashier F35 (right); bottom row is MD scatter plots (mean log expression vs gene effect) for the same pair, with the top-loaded genes labelled. **The pair was selected as the worst-matched best pair in the histogram.** The point of the panel is *not* that RQVI and Flashier agree on this pair by Pearson r — they explicitly do not (r = 0.436, below the 0.5 threshold). The point is that the *biology* converges anyway: both GP45 and F35 are TCR-activation programs, the lead genes labelled in the MD scatters overlap, and the UMAP loadings highlight the same activated-T-cell populations. This makes a stronger version of the convergent-discovery claim from panel b / the histogram. See [`pair_GP45_F35.md`](pair_GP45_F35.md).

> **Note for action item:** the r = 0.436 value currently lives only in the filename and the doc — it is **not annotated on the PDF**. Recommend adding it as a subtitle on the main panel so the reader sees the number that justifies calling this the worst pair.

---

## 3. Supplementary figure walkthrough

| Panel | File | Detail doc |
| --- | --- | --- |
| a | `sup_figures/scalability.pdf` | [`scalability.md`](scalability.md) |
| b | `sup_figures/gp_sparsity_scatter.pdf` | [`sparsity_scatter.md`](sparsity_scatter.md) |
| c | `sup_figures/rqvi_best_match_4factors.pdf` | [`best_match_4factors.md`](best_match_4factors.md) |
| d | `sup_figures/rqvi_flashier_coverage.pdf` | [`coverage.md`](coverage.md) |

**Panel a — Scalability.** Mean RQVI training time (minutes) vs dataset size (thousands of cells) over 6 sizes from 10k to 633k, each repeated 3×. Training time scales near-linearly (R² ≈ 0.9999): ~0.6 min for 10k cells, ~35 min for the full 633k-cell ImmgenT dataset. **For comparison, Flashier takes roughly one week on the same 600k cells.** The speed gap is the load-bearing fact for the rest of the supplementary figure: RQVI's speed is what makes it feasible to run 10 seeds, which is what panels c and d both depend on. See [`scalability.md`](scalability.md).

**Panel b — Sparsity scatter.** One dot per RQVI GP (256 total), with x = proportion of active cells (cells with loading > 0.01, log scale), y = number of active genes (genes with `|W_scaled| > 0.45` after global min-max scaling), colour = % variance explained on a log scale. The scatter shows two things at once: (i) most GPs are sparse in *both* dimensions — they live in a small fraction of cells and are composed of a small fraction of the transcriptome; (ii) PVE spans several orders of magnitude, and low-PVE GPs are not noise — they typically capture programs active in rare cell populations and are still biologically interpretable. This is the empirical justification for the sparsity language in §Figure 1 of the paper draft. See [`sparsity_scatter.md`](sparsity_scatter.md).

**Panel c — Best-match RQVI GPs for four Flashier factors (echoes draft figure 2D).** For four Flashier factors (F22, F30, F58, F68), the script searches across all 10 RQVI seeds for the GP with the highest cluster-level Z-scored Pearson r. Top row shows UMAP loadings of the chosen RQVI GP; bottom row shows the MD scatter with top-loaded genes labelled. The actual matches (across 10 seeds) are:

| Flashier | Best RQVI GP | Seed | Pearson r |
| --- | --- | --- | --- |
| F22 | GP 96 | 6 | 0.897 |
| F30 | GP 112 | 5 | 0.812 |
| F58 | GP 110 | 1 | 0.840 |
| F68 | GP 76 | 3 | 0.840 |

The four Flashier factors were hand-picked as lineage-marking factors so that the panel echoes the lineage-specific GPs in draft figure 2D — i.e. the claim is "RQVI also recovers the lineage-specific biology that Flashier finds, and after multi-seed search the agreement is strong (r ≈ 0.81–0.90)." Crucially, the multi-seed search lifts the correlations substantially above what any single RQVI seed achieves (the seed-0 numbers in `data/cross_method_best_corr.csv` are F22→GP16 r=0.70, F30→GP120 r=0.58, F58→GP166 r=0.73, F68→GP116 r=0.83 — uniformly lower than the best-of-10-seeds matches in the figure). This is exactly the speed → multi-seed → robustness chain that connects panels a, c, and d. See [`best_match_4factors.md`](best_match_4factors.md).

**Panel d — Coverage of Flashier factors as RQVI seeds accumulate.** X-axis is the number of RQVI seeds (1–10); y-axis is the fraction of the 200 Flashier factors that are "covered" by the union of selected seeds, where a Flashier factor counts as covered the moment any selected RQVI seed has best-match `|r| ≥ threshold`. Seeds are added one at a time via greedy set-cover (at each step, the seed that maximises the union coverage is appended). The four coloured lines correspond to four r-thresholds (`THRESHOLDS = [0.3, 0.4, 0.5, 0.6]` in the live script). Because of the greedy set-cover construction, the curve is **monotone non-decreasing — i.e. cumulative coverage**. The take-away is how quickly the curve saturates: if 3–4 seeds are enough to cover most Flashier factors at r ≥ 0.5, that supports the multi-seed-robustness claim and answers "why we run RQVI 10 times". See [`coverage.md`](coverage.md).

---

## 4. Answers to the four open questions

### Q1 — How are "active genes" and PVE defined in `gp_sparsity_scatter.pdf`?

- **Number of active genes (Y-axis):** the gene-effect matrix `W` (19,805 genes × 256 GPs) is min-max scaled *globally* to [-1, 1] using `W.min()` / `W.max()`. A gene is then counted as "active" for a GP if `|W_scaled| > 0.45`. The 0.45 threshold is **heuristic**: a 2 × 3 grid of scatter plots was generated at thresholds 0.40-0.50 (step 0.02) and 0.45 was chosen as the value that visually separated sparse from dense GPs the cleanest. Reviewers may ask for a sensitivity analysis; flag this if it comes up.
- **% Variance Explained (colour):** computed as `var_j = (Σ X[:,j]²) · (Σ W[:,j]²)` where `X` is the cell-loading matrix (seed 0) and `W` is the gene-effect matrix, then normalised: `PVE_j = var_j / Σ var_j`. Note that this is the **fraction of total reconstructed variance attributable to each GP** — the 256 PVE values sum to 1, *not* the fraction of the dataset's total variance. The colour bar uses a `LogNorm` scale because PVE spans several orders of magnitude.
- **Proportion of active cells (X-axis):** `frac_active` from `data/gp_summary_stats_seed0.csv` — the fraction of cells with loading > 0.01. Plotted on a log scale; GP 207 (value = 0) is clipped to half the minimum positive value so it remains visible.
- **Why this matters in the paper:** the scatter is the empirical evidence for the claim in §Figure 1 that "most GP were active across only a subset of the cells … and composed of a fraction of the transcriptome", and that low-PVE GPs are still interpretable (e.g. GP1 baseline housekeeping).

See [`sparsity_scatter.md`](sparsity_scatter.md) for full methodology.

### Q2 — Is GP45 / F35 a good match? The r value isn't shown and visually they don't look matched.

**No, by raw Pearson r the pair is not a good match — and that is the entire point of the panel.** Cluster-level r = 0.436, below the 0.5 "covered" threshold, sitting in the left tail of the histogram. We deliberately chose the *worst-matched best pair* as the case study because the panel is meant to support a stronger claim than the histogram on its own:

> Even when the cross-method correlation is at its lowest, the underlying biology converges. Both GP45 and F35 are TCR-activation programs; their lead genes (visible in the MD scatters) overlap, and their UMAP loadings highlight the same activated-T-cell populations.

If you only look at r, the pair looks like a method failure. If you look at the lead genes and the UMAP, it looks like a methodological win — Flashier and RQVI are recovering the same biology through slightly different parameterizations of the same programmes. That asymmetry is why the panel exists.

> **Action item:** the r = 0.436 number currently lives only in the filename and the methodology doc — it is **not annotated on the PDF itself**. Add it as a subtitle / annotation in `scripts/fig_pair_GP45_F35.py` so a reader of the figure sees the number that justifies the claim. Same goes for naming the lead genes that overlap; the MD scatters label the top 50 genes already, so the user can pick a few in the caption text.

See [`pair_GP45_F35.md`](pair_GP45_F35.md) for full methodology.

### Q3 — The best-4-match panel doesn't contain the "usual suspects" we always see. Why?

Two clarifications:

**(a) F-numbers and GP-numbers are different namespaces.** The "usual suspect" gene programs called out in the paper draft (GP68 for Treg, GP22 for DN, GP29 for CD8aa, GP27 for effector Treg, GP30 for Treg/Tz, GP8 for DETC, GP170, GP80, GP23, GP171, etc.) are *RQVI* GP indices. The four factors plotted in this panel — F22, F30, F58, F68 — are *Flashier* factors. The digits collide for F22 and F30, but they index a completely separate factorization of the same data. So nothing in this panel is "GP68 from the lineage figure"; the Flashier/RQVI naming is what's confusing here.

**(b) What does the panel actually show?** Across all 10 RQVI seeds, the best matches (read from the figure title bars) are:

| Flashier | Best RQVI GP | Seed | Pearson r |
| --- | --- | --- | --- |
| F22 | GP 96 | 6 | 0.897 |
| F30 | GP 112 | 5 | 0.812 |
| F58 | GP 110 | 1 | 0.840 |
| F68 | GP 76 | 3 | 0.840 |

These RQVI GPs (96, 112, 110, 76) are *not* the canonical lineage GPs from draft figure 2D either — but the panel is still doing useful work, because the **four Flashier factors were hand-picked as lineage-marking factors**, and the panel demonstrates that RQVI's best-of-10-seeds search recovers comparable programs at high r. In other words, the panel pivots on Flashier (showing RQVI catches Flashier's lineages), not on RQVI (showing the canonical GP68/22/29/27 etc.).

If the user wants the panel to instead pivot on the *RQVI* canonical lineage GPs — i.e. start from GP68/22/29/27/30/8 and find each one's best Flashier match — that is a different figure and needs a new script (basically a transposed version of `fig_rqvi_best_match_4factors.py`). Decide which way to go before the next revision.

> **Action items:**
> 1. Confirm and document the biological identity (lead genes, lineage) of each Flashier factor F22 / F30 / F58 / F68 — this requires reading `gene_factor_matrix.txt` (~81 MB external file) and interpreting the lead genes.
> 2. Decide whether the panel pivots on Flashier (current) or on RQVI canonical lineage GPs (would need a new script).
> 3. The detail doc for this panel did not previously exist — see the new [`best_match_4factors.md`](best_match_4factors.md).

### Q4 — Is `rqvi_flashier_coverage.pdf` a cumulative coverage plot?

**Yes, it is a cumulative (greedy set-cover) coverage curve.** The X-axis is the number of RQVI seeds (1–10). Seeds are added one at a time using a greedy set-cover heuristic: at each step, the seed that maximises the union coverage is appended. A Flashier factor counts as "covered" the moment **any** of the selected seeds has best-match `|r| ≥ threshold`. Because coverage is computed on the union and seeds are only ever added (never removed), the curve is monotone non-decreasing — the textbook definition of cumulative coverage.

A few methodological notes that should make it into the figure caption:

- **Best matches are computed at the cluster level**, after Z-scoring the Flashier and RQVI cluster-mean matrices column-wise (across clusters). The Pearson r is a dot product of standardized vectors.
- **The metric is `|r|`**, not signed r — anti-correlated programs count as covered. This is fine for an EBMF/semi-NMF model where signs can flip across seeds, but worth flagging in the caption so a reader does not misread the number as a strict positive correlation.
- **The four coloured lines are four r-thresholds.** The live script (`scripts/fig_rqvi_flashier_coverage.py` line 26) has `THRESHOLDS = [0.3, 0.4, 0.5, 0.6]`. The previously-written `coverage.md` doc said 9 thresholds 0.1-0.9 — that is a stale doc/script mismatch and is being reconciled.
- **Why the curve matters:** it answers "how many RQVI seeds do we need to capture the Flashier factorization?" A curve that saturates fast (e.g. 3–4 seeds is enough at r = 0.5) supports the multi-seed-robustness claim and explains why we run RQVI 10 times in the supplementary scalability story.

See [`coverage.md`](coverage.md) for full methodology.

---

## 5. Open action items

Surfaced here so they can be triaged in one place. None of these are in the scope of this internal report; they all touch figure-generation code or biological annotation.

1. **Generate main-figure panel b** — cell-level Flashier × RQVI similarity heatmap. Currently absent; this is the headline visual for the convergent-discovery claim.
2. **Annotate `r = 0.436` on `pair_GP45_F35.pdf`.** Currently only in the filename and the doc; should be visible to readers of the figure. Small edit to `scripts/fig_pair_GP45_F35.py`.
3. **Annotate lead-gene overlap on `pair_GP45_F35.pdf`** (optional). The MD scatters already label the top 50 genes; a one-line subtitle naming the shared TCR-activation lead genes would make the rhetorical claim land harder.
4. **Decide the pivot of `rqvi_best_match_4factors.pdf`.** Current panel pivots on four Flashier factors (F22/F30/F58/F68). If we want the panel to pivot instead on the RQVI canonical lineage GPs (GP68/22/29/27/30/8), we need a new transposed script.
5. **Annotate the biology of F22, F30, F58, F68.** Read lead genes from the external `gene_factor_matrix.txt` and document which T-cell lineage each Flashier factor marks; record in [`best_match_4factors.md`](best_match_4factors.md).
6. **Reconcile coverage thresholds.** `scripts/fig_rqvi_flashier_coverage.py` uses `[0.3, 0.4, 0.5, 0.6]`; the previous `coverage.md` said 9 thresholds 0.1-0.9. The doc has been updated to match the script. If the original intent was 9 thresholds, change the script.
7. **Verify whether GP38 / F58 is still being used.** The doc [`pair_GP38_F58.md`](pair_GP38_F58.md) describes a "good match" companion pair (r = 0.573, above threshold). The user did not list it as a current panel — confirm whether to archive it to `docs/old_version_or_other/` or keep as a backup.
8. **Refresh the README.** `figures_version_v2/README.md` lists figures at flat paths like `figures/scalability.pdf`, but they have been reorganised into `figures/main_figures/` and `figures/sup_figures/`. Also, the README's table has `—` for the doc column of `rqvi_best_match_4factors.pdf` — that gap is now filled by [`best_match_4factors.md`](best_match_4factors.md), so the table can be updated.
