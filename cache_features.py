from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

print("[startup] Loading Python, PyTorch and graph dependencies...", flush=True)

import pandas as pd
from tqdm.auto import tqdm

from rxncontext_common import (
    DEFAULT_DATA_PATH,
    SPLIT_COLUMNS,
    atomic_torch,
    dry_run_subset,
    feature_root_for_run,
    graph_path,
    load_source_data,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache each edge-v4 graph once by reaction ID and reuse it across all splits/seeds."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=SPLIT_COLUMNS, default=list(SPLIT_COLUMNS))
    parser.add_argument("--graph-workers", type=int, default=max(1, min(20, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-samples-per-partition-class", type=int, default=2)
    return parser.parse_args()


def _cache_graph(task: tuple[str, int, str, bool]) -> tuple[str, str]:
    sample_id, label, mapped_reaction, force = task
    from rxncontext_common import graph_path
    from src.graph.graph_dataset_edge_v4 import ReactionGraphEdgeV4Dataset

    root = _WORKER_FEATURE_ROOT
    path = graph_path(root, sample_id)
    if not force and path.is_file():
        return sample_id, "cached"

    frame = pd.DataFrame({"MappedReaction": [mapped_reaction], "Label": [label]})
    graph = ReactionGraphEdgeV4Dataset(frame, label_col="Label")[0]
    atomic_torch(path, graph)
    return sample_id, "built"


_WORKER_FEATURE_ROOT = ""


def _init_graph_worker(feature_root: str) -> None:
    global _WORKER_FEATURE_ROOT
    _WORKER_FEATURE_ROOT = feature_root


def main() -> None:
    args = parse_args()
    if args.graph_workers < 1:
        raise ValueError("--graph-workers must be positive.")
    feature_root = feature_root_for_run(args.dry_run, args.feature_root)
    print(
        f"[1/2] Reading and validating dataset: {args.data_path.resolve()}",
        flush=True,
    )
    frame = load_source_data(args.data_path, require_reactions=True)
    if args.dry_run:
        frame = dry_run_subset(frame, args.splits, args.dry_samples_per_partition_class)
    print(
        f"[1/2] Dataset ready: {len(frame):,} unique reactions.",
        flush=True,
    )
    feature_root.mkdir(parents=True, exist_ok=True)

    tasks = (
        (str(row.ID), int(row.Label), str(row.MappedReaction), bool(args.force))
        for row in frame.itertuples(index=False)
    )
    print(
        f"[2/2] Building edge-v4 caches with {args.graph_workers} threads: "
        f"{feature_root}",
        flush=True,
    )
    with ThreadPoolExecutor(
        max_workers=args.graph_workers,
        initializer=_init_graph_worker,
        initargs=(str(feature_root),),
    ) as pool:
        for _ in tqdm(
            pool.map(_cache_graph, tasks),
            total=len(frame),
            desc="Edge-v4 graphs",
            unit="reaction",
            dynamic_ncols=True,
            file=sys.stderr,
        ):
            pass

    write_manifest(feature_root, frame)
    metadata: dict[str, Any] = {
        "data_file": args.data_path.name,
        "n_reactions": len(frame),
        "dry_run": bool(args.dry_run),
        "graph_storage": "one edge-v4 graph per reaction ID",
    }
    from rxncontext_common import atomic_json

    atomic_json(feature_root / "cache_metadata.json", metadata)
    print(f"Cached {len(frame):,} reactions under: {feature_root}")


if __name__ == "__main__":
    main()
