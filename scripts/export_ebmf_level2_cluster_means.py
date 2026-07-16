"""Aggregate EBMF/Flashier cell loadings over the 114 level2 clusters.

The 2.7 GB Flashier loading matrix contains more cells than the RQVI analysis.
This exporter streams the text file, retains the cells present in the bundled
RQVI exchange file, and calculates an arithmetic mean for every cluster and
every EBMF factor without loading the complete matrix into memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EBMF_CELL = Path(
    "/homes/gws/tianzew/projects/gene_program_model/"
    "Evaluation/Subcluster/cell_factor_matrix.txt"
)
DEFAULT_RQVI_ARCHIVE = PROJECT_DIR / "data" / "rqvi_cell_loadings_seed0.h5"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "ebmf_mean_loadings_by_level2_cluster.csv"
DEFAULT_COUNTS_OUTPUT = PROJECT_DIR / "data" / "ebmf_level2_cluster_counts.csv"


def _decode(dataset: h5py.Dataset) -> np.ndarray:
    return dataset.asstr()[:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ebmf-cell-loadings", type=Path, default=DEFAULT_EBMF_CELL)
    parser.add_argument("--rqvi-archive", type=Path, default=DEFAULT_RQVI_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--counts-output", type=Path, default=DEFAULT_COUNTS_OUTPUT)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=10_000,
        help="Rows read from the Flashier matrix at a time (default: 10000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ebmf_cell_loadings.exists():
        raise FileNotFoundError(args.ebmf_cell_loadings)
    if not args.rqvi_archive.exists():
        raise FileNotFoundError(args.rqvi_archive)
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")

    with h5py.File(args.rqvi_archive, "r") as archive:
        metadata = archive["metadata"]
        reference_cells = _decode(metadata["cell_names"])
        reference_cluster_codes = metadata["level2_cluster_codes"][:].astype(np.int64)
        cluster_labels = _decode(metadata["level2_cluster_labels"])

    if len(reference_cells) != len(reference_cluster_codes):
        raise ValueError("RQVI cell names and cluster codes have different lengths")
    reference_index = pd.Index(reference_cells)
    if not reference_index.is_unique:
        raise ValueError("RQVI archive contains duplicate cell names")

    header = pd.read_csv(args.ebmf_cell_loadings, sep="\t", nrows=0)
    factor_names = header.columns.astype(str).tolist()
    expected_factor_names = [f"F{index}" for index in range(1, 201)]
    if factor_names != expected_factor_names:
        raise ValueError(
            "Unexpected EBMF factor columns; expected F1 through F200, got "
            f"{factor_names[:3]} ... {factor_names[-3:]}"
        )

    n_clusters = len(cluster_labels)
    n_factors = len(factor_names)
    sums = np.zeros((n_clusters, n_factors), dtype=np.float64)
    matched_counts = np.zeros(n_clusters, dtype=np.int64)
    seen_reference_cells = np.zeros(len(reference_cells), dtype=bool)
    total_rows = 0
    unmatched_rows = 0

    print(f"Streaming EBMF loadings from {args.ebmf_cell_loadings}")
    reader = pd.read_csv(
        args.ebmf_cell_loadings,
        sep="\t",
        index_col=0,
        chunksize=args.chunk_rows,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        if chunk.columns.astype(str).tolist() != factor_names:
            raise ValueError(f"Factor columns changed in chunk {chunk_number}")

        total_rows += len(chunk)
        reference_positions = reference_index.get_indexer(chunk.index.astype(str))
        keep = reference_positions >= 0
        unmatched_rows += int((~keep).sum())
        if not np.any(keep):
            continue

        reference_positions = reference_positions[keep]
        if np.any(seen_reference_cells[reference_positions]):
            raise ValueError("Duplicate reference cell encountered in EBMF matrix")
        seen_reference_cells[reference_positions] = True

        cluster_codes = reference_cluster_codes[reference_positions]
        values = chunk.to_numpy(dtype=np.float64, copy=False)[keep]

        row_order = np.argsort(cluster_codes, kind="stable")
        sorted_codes = cluster_codes[row_order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
        unique_codes = sorted_codes[starts]
        sums[unique_codes] += np.add.reduceat(values[row_order], starts, axis=0)
        matched_counts += np.bincount(cluster_codes, minlength=n_clusters)

        if chunk_number == 1 or total_rows % 100_000 < args.chunk_rows:
            print(
                f"  read {total_rows:,} EBMF rows; "
                f"matched {seen_reference_cells.sum():,}/{len(reference_cells):,}",
                flush=True,
            )

    missing_reference = int((~seen_reference_cells).sum())
    if missing_reference:
        raise ValueError(
            f"EBMF matrix is missing {missing_reference:,} cells from the RQVI archive"
        )
    if np.any(matched_counts == 0):
        empty = cluster_labels[matched_counts == 0].tolist()
        raise ValueError(f"No matched EBMF cells for clusters: {empty}")

    means = sums / matched_counts[:, None]
    mean_df = pd.DataFrame(means, index=cluster_labels, columns=factor_names)
    mean_df.index.name = "level2_cluster"
    count_df = pd.DataFrame(
        {
            "level2_cluster": cluster_labels,
            "n_common_cells": matched_counts,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.counts_output.parent.mkdir(parents=True, exist_ok=True)
    mean_df.to_csv(args.output, float_format="%.17g")
    count_df.to_csv(args.counts_output, index=False)

    print(f"Saved EBMF cluster means: {args.output}")
    print(f"Saved EBMF cluster counts: {args.counts_output}")
    print(
        f"Input rows={total_rows:,}; common cells={seen_reference_cells.sum():,}; "
        f"EBMF-only rows={unmatched_rows:,}; clusters={n_clusters}; factors={n_factors}"
    )


if __name__ == "__main__":
    main()
