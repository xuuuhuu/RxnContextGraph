"""Build split/partition matrices for the gat_edge_v4 context input."""
from __future__ import annotations

import argparse
from pathlib import Path

from context_pipeline.ml_common import (
    DEFAULT_FEATURE_ROOT,
    SPLIT_COLUMNS,
    ensure_matrix_caches,
    read_cache_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--splits", nargs="+", choices=SPLIT_COLUMNS, default=list(SPLIT_COLUMNS))
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cache_root = args.cache_root.expanduser().resolve()
    manifest = read_cache_manifest(cache_root)
    ensure_matrix_caches(
        cache_root,
        manifest,
        splits=args.splits,
        features=("rxnfp_mapped_v2",),
        workers=args.workers,
        chunk_size=args.chunk_size,
        force=args.force,
    )
    print(f"Built gat_edge_v4 context matrices under {cache_root}")


if __name__ == "__main__":
    main()
