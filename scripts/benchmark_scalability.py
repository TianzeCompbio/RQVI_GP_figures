"""
Benchmark RQVI training time vs. number of cells.

Subsamples the immgenT dataset at 6 sizes, trains RQVI 3 times each,
and records wall-clock training time to CSV.

Usage:
    python figures_version_v2/scripts/benchmark_scalability.py --device 3
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import csr_matrix
from lightning.pytorch.callbacks import Callback
from pathlib import Path

torch.set_float32_matmul_precision("medium")

# --- Paths ----------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
RQVI_DIR = "/homes/gws/tianzew/projects/gene_program_model/scVI_model/RQVI"
ADATA_PATH = "/data/tianzew/immgenT/david_final_10k_genes.h5ad"
OUTPUT_CSV = PROJECT_DIR / "data/scalability_benchmark.csv"

# --- Import RQVI ---------------------------------------------------------
sys.path.insert(0, RQVI_DIR)
from mymodel import MyModel

# --- Constants ------------------------------------------------------------
SUBSAMPLE_SIZES = [10_000, 50_000, 100_000, 200_000, 400_000, 633_684]
N_REPEATS = 3
FULL_N_CELLS = 633_684
BASE_PROGRESSIVE_STEPS = [0, 4000, 6000, 8000]


# --- Callback for progressive schedule ------------------------------------
class ProgressiveScheduleCallback(Callback):
    """Update RQ-VAE depth on every training batch end."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if hasattr(pl_module, "module"):
            pl_module.module.update_progressive_schedule(trainer.global_step)


def scale_progressive_steps(n_cells: int) -> list:
    """Scale progressive steps proportionally to dataset size."""
    ratio = n_cells / FULL_N_CELLS
    return [int(s * ratio) for s in BASE_PROGRESSIVE_STEPS]


def stratified_subsample(adata, n_cells: int, seed: int) -> sc.AnnData:
    """Subsample preserving level1 cell-type proportions."""
    if n_cells >= adata.n_obs:
        return adata.copy()

    rng = np.random.default_rng(seed)
    indices = []
    labels = adata.obs["level1"].values
    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_idx = np.where(labels == label)[0]
        n_take = max(1, int(round(n_cells * len(label_idx) / adata.n_obs)))
        chosen = rng.choice(label_idx, size=min(n_take, len(label_idx)), replace=False)
        indices.append(chosen)

    indices = np.concatenate(indices)
    # Adjust to exact target size
    if len(indices) > n_cells:
        indices = rng.choice(indices, size=n_cells, replace=False)
    elif len(indices) < n_cells:
        remaining = np.setdiff1d(np.arange(adata.n_obs), indices)
        extra = rng.choice(remaining, size=n_cells - len(indices), replace=False)
        indices = np.concatenate([indices, extra])

    sub = adata[indices].copy()
    # Remove unused batch categories to avoid NaN in library size init
    for col in sub.obs.select_dtypes(include="category").columns:
        sub.obs[col] = sub.obs[col].cat.remove_unused_categories()
    return sub


def run_benchmark(adata_full, n_cells, repeat, device):
    """Train RQVI on a subsample and return wall-clock seconds."""
    seed = repeat * 100
    print(f"\n{'='*60}")
    print(f"  n_cells={n_cells:,}  repeat={repeat}  seed={seed}")
    print(f"{'='*60}")

    # Subsample
    adata = stratified_subsample(adata_full, n_cells, seed=seed)
    print(f"  Subsampled: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Ensure counts layer exists and X is CSR (matching runner_wandb.py)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    if not isinstance(adata.layers["counts"], csr_matrix):
        adata.layers["counts"] = csr_matrix(adata.layers["counts"])

    # Scale progressive steps
    prog_steps = scale_progressive_steps(n_cells)
    print(f"  Progressive steps: {prog_steps}")

    # Setup anndata
    MyModel.setup_anndata(adata, layer="counts", batch_key="IGT")

    # Create model
    model = MyModel(
        adata,
        n_hidden=512,
        n_latent=256,
        n_vectors_unsupervised=64,
        n_layers=2,
        max_depth=4,
        commitment_loss_coef=0.8,
        ema_decay=0.99,
        restart_unused_codes=True,
        progressive_steps=prog_steps,
    )

    # Train with timing
    callbacks = [ProgressiveScheduleCallback()]

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    model.train(
        devices=[device],
        max_epochs=30,
        batch_size=256,
        train_size=0.9,
        logger=False,
        callbacks=callbacks,
    )

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    print(f"  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="RQVI scalability benchmark")
    parser.add_argument("--device", type=int, default=3, help="GPU device index")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # After restricting visibility, device index becomes 0
    device_idx = 0

    print(f"Using GPU: {args.device} (CUDA_VISIBLE_DEVICES={args.device})")
    print(f"Loading full dataset from {ADATA_PATH} ...")
    adata_full = sc.read_h5ad(ADATA_PATH)
    print(f"  Full dataset: {adata_full.n_obs:,} cells x {adata_full.n_vars:,} genes")

    # Load existing results if present (for resuming)
    if OUTPUT_CSV.exists():
        results = pd.read_csv(OUTPUT_CSV).to_dict("records")
        done = {(r["n_cells"], r["repeat"]) for r in results}
        print(f"Resuming: {len(results)} runs already completed")
    else:
        results = []
        done = set()

    for n_cells in SUBSAMPLE_SIZES:
        for repeat in range(N_REPEATS):
            if (n_cells, repeat) in done:
                print(f"Skipping n_cells={n_cells:,} repeat={repeat} (already done)")
                continue

            elapsed = run_benchmark(adata_full, n_cells, repeat, device_idx)

            results.append({
                "n_cells": n_cells,
                "repeat": repeat,
                "time_seconds": elapsed,
                "time_minutes": elapsed / 60,
            })

            # Save incrementally
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            print(f"  Saved to {OUTPUT_CSV}")

    print("\nAll benchmarks complete!")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
