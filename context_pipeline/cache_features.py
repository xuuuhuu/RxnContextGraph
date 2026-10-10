from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm.auto import tqdm

from context_pipeline.ml_common import (
    DEFAULT_DATA_PATH,
    FEATURES,
    FEATURE_DIMS,
    MAPPED_FEATURE_MODE,
    PARTITIONS,
    SPLIT_COLUMNS,
    atomic_json,
    atomic_save_feature,
    build_mapped_featurizer,
    build_rxnfp_generator,
    cache_root_for_run,
    encode_base_features,
    encode_rxnfp_reactions,
    load_feature_vector,
    load_metadata,
    sample_cache_path,
    source_file_signature,
    vector_at,
)


BASE_PHASE_ORDER = ("rxnfp", "drfp", "mapped_v2")
HYBRID_PHASE_ORDER = ("drfp_mapped_v2", "rxnfp_mapped_v2")
ALL_PHASE_ORDER = (*BASE_PHASE_ORDER, *HYBRID_PHASE_ORDER)

_FEATURE_WORKER_NAME = ""
_MAPPED_FEATURIZER: Any | None = None
_RXNFP_GENERATOR: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate feature caches in sequential phases; each phase runs its own worker pool."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--dry-run", action="store_true", help="Use isolated feature/.dry_run cache.")
    parser.add_argument(
        "--dry-run-per-cell",
        type=int,
        default=2,
        help="Rows sampled per split × partition × label cell in dry-run mode.",
    )
    parser.add_argument("--batch-size", type=int, default=2048, help="CSV rows read per feature phase.")
    parser.add_argument("--rxnfp-batch-size", type=int, default=16)
    parser.add_argument("--rxnfp-workers", type=int, default=1)
    parser.add_argument("--rxnfp-gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--cpu-workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=None)
    return parser.parse_args()


def dry_run_sample(metadata: pd.DataFrame, per_cell: int, seed: int) -> pd.DataFrame:
    if per_cell < 1:
        raise ValueError("--dry-run-per-cell must be >= 1")
    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    for split_col in SPLIT_COLUMNS:
        for partition in PARTITIONS:
            for label in (0, 1):
                ids = metadata.loc[
                    (metadata[split_col] == partition) & (metadata["Label"] == label), "ID"
                ].to_numpy(dtype=str)
                if len(ids):
                    selected.update(rng.choice(ids, size=min(per_cell, len(ids)), replace=False).tolist())
    sample = metadata.loc[metadata["ID"].isin(selected)].copy()
    if sample.empty:
        raise ValueError("Could not select any rows for dry run.")
    return sample.reset_index(drop=True)


def _initialize_feature_worker(feature_name: str) -> None:
    global _FEATURE_WORKER_NAME, _MAPPED_FEATURIZER
    from threadpoolctl import threadpool_limits

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    threadpool_limits(limits=1)
    _FEATURE_WORKER_NAME = feature_name
    _MAPPED_FEATURIZER = (
        build_mapped_featurizer(MAPPED_FEATURE_MODE) if feature_name == "mapped_v2" else None
    )


def _encode_feature_partition(frame: pd.DataFrame) -> np.ndarray | sp.csr_matrix:
    if _FEATURE_WORKER_NAME == "drfp":
        result = encode_base_features(
            frame,
            want_drfp=True,
            want_rxnfp=False,
            want_mapped=False,
            rxnfp_generator=None,
            mapped_featurizer=None,
        )
    elif _FEATURE_WORKER_NAME == "mapped_v2":
        result = encode_base_features(
            frame,
            want_drfp=False,
            want_rxnfp=False,
            want_mapped=True,
            rxnfp_generator=None,
            mapped_featurizer=_MAPPED_FEATURIZER,
        )
    else:
        raise RuntimeError(f"Unexpected CPU feature phase: {_FEATURE_WORKER_NAME}")
    return result[_FEATURE_WORKER_NAME]


def _initialize_rxnfp_worker(gpu_id: int) -> None:
    global _RXNFP_GENERATOR
    _RXNFP_GENERATOR = build_rxnfp_generator(gpu_id=gpu_id)


def _encode_rxnfp_partition(reactions: list[str], batch_size: int) -> np.ndarray:
    if _RXNFP_GENERATOR is None:
        raise RuntimeError("RXNFP worker was not initialized.")
    return encode_rxnfp_reactions(_RXNFP_GENERATOR, reactions, batch_size=batch_size)


def _split_frame(frame: pd.DataFrame, count: int) -> list[pd.DataFrame]:
    part_count = min(count, len(frame))
    boundaries = np.linspace(0, len(frame), part_count + 1, dtype=int)
    return [frame.iloc[start:stop].reset_index(drop=True) for start, stop in zip(boundaries[:-1], boundaries[1:])]


def _cpu_encode_batch(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    worker_count: int,
    pool: ProcessPoolExecutor,
    on_progress: Callable[[int], None],
) -> np.ndarray | sp.csr_matrix:
    parts = _split_frame(frame, worker_count)
    _ = feature_name
    futures = {pool.submit(_encode_feature_partition, part): i for i, part in enumerate(parts)}
    matrices: list[np.ndarray | sp.csr_matrix | None] = [None] * len(parts)
    for future in as_completed(futures):
        i = futures[future]
        matrices[i] = future.result()
        on_progress(len(parts[i]))
    complete_matrices = [matrix for matrix in matrices if matrix is not None]
    if sp.issparse(complete_matrices[0]):
        return sp.vstack(complete_matrices, format="csr", dtype=np.float32)
    return np.concatenate(complete_matrices, axis=0).astype(np.float32, copy=False)


def _rxnfp_encode_batch(
    frame: pd.DataFrame,
    *,
    gpu_ids: list[int],
    worker_count: int,
    batch_size: int,
    pools: dict[int, ProcessPoolExecutor],
    worker_rotation: int,
    on_progress: Callable[[int], None],
) -> np.ndarray:
    reactions = frame["Reaction"].fillna("").astype(str).tolist()
    worker_counts = [worker_count // len(gpu_ids)] * len(gpu_ids)
    for i in range(worker_count % len(gpu_ids)):
        worker_counts[i] += 1
    slots = [gpu_id for gpu_id, count in zip(gpu_ids, worker_counts) for _ in range(count)]
    reaction_parts = _split_values(reactions, worker_count)

    futures = {
        pools[slots[(worker_rotation + i) % len(slots)]].submit(
            _encode_rxnfp_partition, part, batch_size
        ): i
        for i, part in enumerate(reaction_parts)
    }
    encoded_parts: list[np.ndarray | None] = [None] * len(reaction_parts)
    for future in as_completed(futures):
        i = futures[future]
        encoded_parts[i] = future.result()
        on_progress(len(reaction_parts[i]))
    return np.concatenate([part for part in encoded_parts if part is not None], axis=0).astype(np.float32, copy=False)


def _split_values(values: list[str], count: int) -> list[list[str]]:
    part_count = min(count, len(values))
    boundaries = np.linspace(0, len(values), part_count + 1, dtype=int)
    return [values[start:stop] for start, stop in zip(boundaries[:-1], boundaries[1:])]


def _combine_one(
    task: tuple[str, str, str, str, str],
) -> str:
    sample_id, feature_a, feature_b, output_feature, cache_root = task
    vector_a = load_feature_vector(cache_root, feature_a, sample_id)
    vector_b = load_feature_vector(cache_root, feature_b, sample_id)
    atomic_save_feature(
        sample_cache_path(cache_root, output_feature, sample_id),
        np.concatenate([vector_a, vector_b]),
    )
    return sample_id


def _missing_ids(
    ids: Iterable[str],
    *,
    cache_root: Path,
    feature_name: str,
    overwrite: bool,
) -> list[str]:
    return [
        sample_id
        for sample_id in ids
        if overwrite or not sample_cache_path(cache_root, feature_name, sample_id).is_file()
    ]


def _run_base_phase(
    feature_name: str,
    ids: list[str],
    *,
    args: argparse.Namespace,
    cache_root: Path,
    source_row_count: int,
    context: mp.context.BaseContext,
) -> None:
    missing = set(_missing_ids(ids, cache_root=cache_root, feature_name=feature_name, overwrite=args.overwrite))
    if not missing:
        print(f"[{feature_name}] all {len(ids):,} ID caches already exist; skipping phase.")
        return

    if feature_name == "rxnfp":
        import torch

        gpu_ids = list(dict.fromkeys(args.rxnfp_gpus))
        invalid = [gpu for gpu in gpu_ids if gpu < 0 or gpu >= torch.cuda.device_count()]
        if not torch.cuda.is_available() or invalid:
            raise RuntimeError(
                f"RXNFP requires visible CUDA devices {gpu_ids}; visible device count={torch.cuda.device_count()}."
            )
        if not gpu_ids or args.rxnfp_workers < len(gpu_ids):
            raise ValueError("RXNFP worker count must be at least the number of selected GPUs.")
        worker_count = args.rxnfp_workers
        worker_counts = [worker_count // len(gpu_ids)] * len(gpu_ids)
        for i in range(worker_count % len(gpu_ids)):
            worker_counts[i] += 1
        distribution = ", ".join(
            f"GPU {gpu_id}: {count}" for gpu_id, count in zip(gpu_ids, worker_counts)
        )
        print(f"[{feature_name}] starting {worker_count} workers ({distribution}).")
    else:
        gpu_ids = []
        worker_count = args.cpu_workers
        print(f"[{feature_name}] starting {worker_count} CPU workers.")

    usecols = ["ID", "Reaction"]
    if feature_name == "mapped_v2":
        usecols.append("MappedReaction")
    total_rows = source_row_count
    row_bar = tqdm(total=total_rows, desc=f"{feature_name}: 扫描 CSV", unit="row", dynamic_ncols=True)
    encode_bar = tqdm(total=len(missing), desc=f"{feature_name}: 特征", unit="rxn", position=1, leave=True, dynamic_ncols=True)
    written = 0
    rxnfp_rotation = 0
    try:
        with ExitStack() as stack:
            cpu_pool = None
            rxnfp_pools: dict[int, ProcessPoolExecutor] = {}
            if feature_name == "rxnfp":
                worker_counts = [worker_count // len(gpu_ids)] * len(gpu_ids)
                for i in range(worker_count % len(gpu_ids)):
                    worker_counts[i] += 1
                rxnfp_pools = {
                    gpu_id: stack.enter_context(
                        ProcessPoolExecutor(
                            max_workers=count,
                            mp_context=context,
                            initializer=_initialize_rxnfp_worker,
                            initargs=(gpu_id,),
                        )
                    )
                    for gpu_id, count in zip(gpu_ids, worker_counts)
                    if count > 0
                }
            else:
                cpu_pool = stack.enter_context(
                    ProcessPoolExecutor(
                        max_workers=worker_count,
                        mp_context=context,
                        initializer=_initialize_feature_worker,
                        initargs=(feature_name,),
                    )
                )

            iterator = pd.read_csv(
                args.data_path,
                usecols=usecols,
                dtype={"ID": "string"},
                chunksize=args.batch_size,
                low_memory=False,
            )
            for frame in iterator:
                chunk_rows = len(frame)
                row_bar.update(chunk_rows)
                frame["ID"] = frame["ID"].astype(str)
                frame = frame.loc[frame["ID"].isin(missing)]
                if not frame.empty:
                    frame = frame.reset_index(drop=True)
                    if feature_name == "rxnfp":
                        encoded = _rxnfp_encode_batch(
                            frame,
                            gpu_ids=gpu_ids,
                            worker_count=worker_count,
                            batch_size=args.rxnfp_batch_size,
                            pools=rxnfp_pools,
                            worker_rotation=rxnfp_rotation,
                            on_progress=encode_bar.update,
                        )
                        rxnfp_rotation += len(frame)
                    else:
                        assert cpu_pool is not None
                        encoded = _cpu_encode_batch(
                            frame,
                            feature_name=feature_name,
                            worker_count=worker_count,
                            pool=cpu_pool,
                            on_progress=encode_bar.update,
                        )
                    for row_idx, sample_id in enumerate(frame["ID"].astype(str)):
                        atomic_save_feature(
                            sample_cache_path(cache_root, feature_name, sample_id),
                            vector_at(encoded, row_idx),
                        )
                        missing.discard(sample_id)
                    written += len(frame)
                row_bar.set_postfix(cached=len(ids) - len(missing), remaining=len(missing))
    finally:
        row_bar.close()
        encode_bar.close()

    if missing:
        raise RuntimeError(f"[{feature_name}] failed to generate {len(missing)} ID cache(s).")
    print(f"[{feature_name}] generated {written:,} cache entries.")


def _run_hybrid_phase(
    feature_name: str,
    ids: list[str],
    *,
    args: argparse.Namespace,
    cache_root: Path,
) -> None:
    feature_a, feature_b = ("drfp", "mapped_v2") if feature_name == "drfp_mapped_v2" else ("rxnfp", "mapped_v2")
    missing = _missing_ids(ids, cache_root=cache_root, feature_name=feature_name, overwrite=args.overwrite)
    if not missing:
        print(f"[{feature_name}] all {len(ids):,} ID caches already exist; skipping phase.")
        return
    missing_set = set(missing)
    absent = [
        f"{base_feature}: {sample_id}"
        for sample_id in missing
        for base_feature in (feature_a, feature_b)
        if not sample_cache_path(cache_root, base_feature, sample_id).is_file()
    ]
    if absent:
        raise FileNotFoundError(
            f"[{feature_name}] prerequisite caches are missing; complete {feature_a} and mapped_v2 first. "
            f"Examples: {absent[:5]}"
        )

    tasks = (
        (sample_id, feature_a, feature_b, feature_name, str(cache_root))
        for sample_id in ids
        if sample_id in missing_set
    )
    with ProcessPoolExecutor(max_workers=args.cpu_workers, mp_context=mp.get_context("spawn")) as pool:
        for _ in tqdm(
            pool.map(_combine_one, tasks, chunksize=16),
            total=len(missing),
            desc=f"{feature_name}: 拼接缓存",
            unit="rxn",
            dynamic_ncols=True,
        ):
            pass
    print(f"[{feature_name}] generated {len(missing):,} cache entries.")


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.rxnfp_batch_size, args.rxnfp_workers, args.cpu_workers) < 1:
        raise ValueError("All batch sizes and worker counts must be >= 1.")
    args.data_path = args.data_path.expanduser().resolve()
    cache_root = cache_root_for_run(args.dry_run, args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.data_path)
    source_row_count = len(metadata)
    if args.dry_run:
        metadata = dry_run_sample(metadata, args.dry_run_per_cell, args.seed)
        print(f"Dry run selected {len(metadata):,} unique reactions across the split/label cells.")
    metadata.to_csv(cache_root / "manifest.csv", index=False)
    ids = metadata["ID"].astype(str).tolist()

    selected = set(args.features)
    ordered_features = [feature for feature in ALL_PHASE_ORDER if feature in selected]
    signature = source_file_signature(args.data_path)
    for feature_name in ordered_features:
        (cache_root / feature_name).mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    for feature_name in ordered_features:
        print(f"\n=== Feature phase: {feature_name} ===")
        if feature_name in BASE_PHASE_ORDER:
            _run_base_phase(
                feature_name,
                ids,
                args=args,
                cache_root=cache_root,
                source_row_count=source_row_count,
                context=context,
            )
        else:
            _run_hybrid_phase(feature_name, ids, args=args, cache_root=cache_root)
        atomic_json(
            cache_root / feature_name / "cache_meta.json",
            {
                "feature": feature_name,
                "dimension": FEATURE_DIMS[feature_name],
                "mapped_feature_mode": MAPPED_FEATURE_MODE if "mapped_v2" in feature_name else None,
                "source": signature,
                "sample_count": len(metadata),
            },
        )
    print(f"All selected feature phases finished under: {cache_root}")


if __name__ == "__main__":
    main()
