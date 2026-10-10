from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from rxncontext_common import feature_root_for_run, graph_path, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack existing edge-v4 .pt files into contiguous arrays for fast, "
            "memory-mapped training reads."
        )
    )
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(20, os.cpu_count() or 1),
        help="Reader threads; deserialization remains single-threaded to avoid GIL contention.",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_graph(task: tuple[Path, str]) -> tuple[str, bytes]:
    feature_root, sample_id = task
    path = graph_path(feature_root, sample_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing graph cache: {path}")
    return sample_id, path.read_bytes()


def _decode_graph(sample_id: str, payload: bytes) -> tuple[str, dict[str, np.ndarray]]:
    data = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    arrays = {
        "x": data.x.detach().numpy().astype(np.float32, copy=False),
        "edge_index": data.edge_index.detach().t().contiguous().numpy().astype(np.int64, copy=False),
        "edge_attr": data.edge_attr.detach().numpy().astype(np.float32, copy=False),
        "center_mask": data.center_mask.detach().numpy().astype(np.float32, copy=False),
        "center_local_mask": data.center_local_mask.detach().numpy().astype(np.float32, copy=False),
    }
    node_count = arrays["x"].shape[0]
    edge_count = arrays["edge_attr"].shape[0]
    if arrays["x"].ndim != 2 or arrays["edge_attr"].ndim != 2:
        raise ValueError(f"Invalid x/edge_attr rank for {sample_id}")
    if arrays["edge_index"].shape != (edge_count, 2):
        raise ValueError(f"edge_index/edge_attr mismatch for {sample_id}")
    if arrays["center_mask"].shape != (node_count,) or arrays[
        "center_local_mask"
    ].shape != (node_count,):
        raise ValueError(f"Node mask shape mismatch for {sample_id}")
    return sample_id, arrays


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError("--workers and --chunk-size must be positive.")
    feature_root = feature_root_for_run(args.dry_run, args.feature_root)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else feature_root / "packed_edge_v4"
    )
    manifest = read_manifest(str(feature_root))
    all_ids = manifest["ID"].astype(str).tolist()
    if args.start < 0 or args.start >= len(all_ids):
        raise ValueError("--start must select a row in the manifest.")
    ids = all_ids[args.start :]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")
        ids = ids[: args.limit]
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Output already exists: {output_root}. Use --force to replace it."
            )
        shutil.rmtree(output_root)
    build_root = output_root.parent / f".{output_root.name}.building.{os.getpid()}"
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)

    handles: dict[str, Any] = {}
    node_offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    edge_offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    node_dim: int | None = None
    edge_dim: int | None = None
    id_hash = hashlib.sha256()
    try:
        for filename in (
            "x.f32",
            "edge_index.i64",
            "edge_attr.f32",
            "center_mask.f32",
            "center_local_mask.f32",
        ):
            handles[filename] = (build_root / filename).open("wb", buffering=8 * 1024 * 1024)
        position = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(
            total=len(ids),
            desc="Packing edge-v4 graphs",
            unit="graph",
            dynamic_ncols=True,
            file=sys.stderr,
        ) as progress:
            for id_chunk in _chunks(ids, args.chunk_size):
                tasks = ((feature_root, sample_id) for sample_id in id_chunk)
                for sample_id, payload in pool.map(_read_graph, tasks):
                    sample_id, arrays = _decode_graph(sample_id, payload)
                    if node_dim is None:
                        node_dim = int(arrays["x"].shape[1])
                        edge_dim = int(arrays["edge_attr"].shape[1])
                    if arrays["x"].shape[1] != node_dim or arrays["edge_attr"].shape[1] != edge_dim:
                        raise ValueError(f"Feature dimension mismatch for {sample_id}")
                    arrays["x"].tofile(handles["x.f32"])
                    arrays["edge_index"].tofile(handles["edge_index.i64"])
                    arrays["edge_attr"].tofile(handles["edge_attr.f32"])
                    arrays["center_mask"].tofile(handles["center_mask.f32"])
                    arrays["center_local_mask"].tofile(handles["center_local_mask.f32"])
                    node_offsets[position + 1] = node_offsets[position] + arrays["x"].shape[0]
                    edge_offsets[position + 1] = edge_offsets[position] + arrays["edge_attr"].shape[0]
                    id_hash.update(sample_id.encode("utf-8"))
                    id_hash.update(b"\0")
                    position += 1
                    progress.update(1)
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()
        np.save(build_root / "node_offsets.npy", node_offsets, allow_pickle=False)
        np.save(build_root / "edge_offsets.npy", edge_offsets, allow_pickle=False)
        metadata = {
            "format": "edge_v4_contiguous_v1",
            "count": len(ids),
            "manifest_count": len(manifest),
            "manifest_start": args.start,
            "complete_manifest": args.start == 0 and len(ids) == len(manifest),
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "total_nodes": int(node_offsets[-1]),
            "total_edges": int(edge_offsets[-1]),
            "ids_sha256": id_hash.hexdigest(),
        }
        (build_root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(build_root, output_root)
    except BaseException:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(build_root, ignore_errors=True)
        raise
    size_gib = sum(path.stat().st_size for path in output_root.iterdir()) / 2**30
    print(
        f"Packed {len(ids):,} graphs ({node_offsets[-1]:,} nodes, "
        f"{edge_offsets[-1]:,} edges) into {output_root} [{size_gib:.2f} GiB]",
        flush=True,
    )


if __name__ == "__main__":
    main()
