"""Export the exact seed-0 RQVI cell loadings in a shareable HDF5 file.

The source AnnData file is large because it contains considerably more than the
cell-loading matrix.  This exporter keeps only what is needed to reproduce the
level2-cluster heatmap:

* the exact cells x programs float32 loading matrix;
* cell names and program names/indices;
* the level2-cluster and level1 annotation for every cell.

The output uses lossless gzip compression and is small enough to keep with the
figure repository.  No thresholding or numeric down-casting is applied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path(
    "/data/tianzew/immgenT/RQVI_multiseeds/results/"
    "cmtloss08_64by4GPs_seed0.h5ad"
)
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "rqvi_cell_loadings_seed0.h5"

LEVEL2_CLUSTER_COL = "Cluster_totalvi20240525rmigtsample_Res0.5"
LEVEL1_COL = "level1"


def _fixed_utf8(values: np.ndarray | pd.Index) -> np.ndarray:
    """Encode strings as a fixed-width byte array that HDF5 can compress."""
    encoded = [str(value).encode("utf-8") for value in values]
    width = max((len(value) for value in encoded), default=1)
    return np.asarray(encoded, dtype=f"S{width}")


def _factorize(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return non-negative codes and sorted labels, retaining literal 'nan'."""
    strings = values.astype(str)
    codes, labels = pd.factorize(strings, sort=True)
    if np.any(codes < 0):
        raise ValueError(f"Missing values remain after factorizing {values.name!r}")
    return codes, np.asarray(labels.astype(str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=2048,
        help="Rows per HDF5 chunk (default: 2048).",
    )
    parser.add_argument(
        "--compression",
        type=int,
        choices=range(1, 10),
        default=9,
        metavar="1-9",
        help="gzip compression level (default: 9).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    if temporary_output.exists():
        temporary_output.unlink()

    print(f"Reading metadata from {source}")
    adata = ad.read_h5ad(source, backed="r")
    try:
        for column in (LEVEL2_CLUSTER_COL, LEVEL1_COL):
            if column not in adata.obs:
                raise KeyError(f"Required obs column not found: {column}")

        n_cells, n_programs = adata.shape
        cell_names = _fixed_utf8(adata.obs_names)
        program_names_as_text = np.asarray(adata.var_names.astype(str))
        program_names = _fixed_utf8(program_names_as_text)
        try:
            program_indices = np.asarray(
                [int(name) for name in program_names_as_text], dtype=np.int16
            )
        except ValueError:
            program_indices = np.arange(n_programs, dtype=np.int16)

        cluster_codes, cluster_labels_text = _factorize(
            adata.obs[LEVEL2_CLUSTER_COL]
        )
        level1_codes, level1_labels_text = _factorize(adata.obs[LEVEL1_COL])
        cluster_labels = _fixed_utf8(cluster_labels_text)
        level1_labels = _fixed_utf8(level1_labels_text)

        # Record the lineage nested above each level2 cluster.  The assertion is
        # useful because the heatmap draws a single lineage strip per cluster.
        n_clusters = len(cluster_labels_text)
        cluster_level1_codes = np.full(n_clusters, -1, dtype=np.int8)
        for cluster_code in range(n_clusters):
            member_level1 = np.unique(level1_codes[cluster_codes == cluster_code])
            if len(member_level1) != 1:
                raise ValueError(
                    f"Cluster {cluster_labels_text[cluster_code]!r} maps to "
                    f"{len(member_level1)} level1 labels"
                )
            cluster_level1_codes[cluster_code] = member_level1[0]

        print(
            f"Exporting {n_cells:,} cells x {n_programs} programs and "
            f"{n_clusters} level2 clusters"
        )
        with h5py.File(temporary_output, "w") as handle:
            handle.attrs["format"] = "RQVI cell-loading exchange file"
            handle.attrs["format_version"] = "1.0"
            handle.attrs["seed"] = 0
            handle.attrs["source_h5ad"] = str(source)
            handle.attrs["level2_cluster_column"] = LEVEL2_CLUSTER_COL
            handle.attrs["level1_column"] = LEVEL1_COL
            handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            handle.attrs["numeric_transform"] = "none; exact source float32 values"

            loading_out = handle.create_dataset(
                "cell_loadings",
                shape=(n_cells, n_programs),
                dtype=np.float32,
                chunks=(min(args.chunk_rows, n_cells), n_programs),
                compression="gzip",
                compression_opts=args.compression,
                shuffle=True,
            )

            metadata = handle.create_group("metadata")
            metadata.create_dataset(
                "cell_names",
                data=cell_names,
                compression="gzip",
                compression_opts=args.compression,
            )
            metadata.create_dataset("program_names", data=program_names)
            metadata.create_dataset("program_indices", data=program_indices)
            metadata.create_dataset(
                "level2_cluster_codes",
                data=cluster_codes.astype(np.int16),
                compression="gzip",
                compression_opts=args.compression,
                shuffle=True,
            )
            metadata.create_dataset("level2_cluster_labels", data=cluster_labels)
            metadata.create_dataset(
                "level1_codes",
                data=level1_codes.astype(np.int8),
                compression="gzip",
                compression_opts=args.compression,
                shuffle=True,
            )
            metadata.create_dataset("level1_labels", data=level1_labels)
            metadata.create_dataset(
                "level2_cluster_level1_codes", data=cluster_level1_codes
            )

            for start in range(0, n_cells, args.chunk_rows):
                stop = min(start + args.chunk_rows, n_cells)
                loading_out[start:stop] = np.asarray(
                    adata.X[start:stop], dtype=np.float32
                )
                if start == 0 or stop == n_cells or stop % 100_000 < args.chunk_rows:
                    print(f"  wrote {stop:,}/{n_cells:,} cells", flush=True)

        temporary_output.replace(output)
    finally:
        adata.file.close()

    size_mib = output.stat().st_size / (1024**2)
    print(f"Saved {output} ({size_mib:.1f} MiB)")


if __name__ == "__main__":
    main()
