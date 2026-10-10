from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack, save_npz, load_npz
from tqdm.auto import tqdm


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_DATA_PATH = PROJECT_DIR / "new_data" / "New_RxnContextGraph.csv"
DEFAULT_FEATURE_ROOT = PROJECT_DIR / "context_feature"
# Resolve the copied featurizer from this release directory.
sys.path.insert(0, str(PROJECT_DIR))


SPLIT_COLUMNS = (
    "random_split",
    "scaffold_split",
    "reaction_core_split",
    "drfp_kmeans100_greedy_split",
)
PARTITIONS = ("train", "val", "test")
FEATURES = (
    "rxnfp",
    "mapped_v2",
    "rxnfp_mapped_v2",
)
N_BITS = 2048
RADIUS = 2
MAPPED_FEATURE_MODE = "reaction_mapped_v2"

FEATURE_DIMS = {
    "drfp": N_BITS,
    "rxnfp": 256,
    "mapped_v2": N_BITS + 2 * N_BITS + 29,
    "drfp_mapped_v2": 2 * N_BITS + 2 * N_BITS + 29,
    "rxnfp_mapped_v2": 256 + N_BITS + 2 * N_BITS + 29,
}

MANIFEST_COLUMNS = ("ID", "Label", *SPLIT_COLUMNS)


def source_file_signature(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    stat = p.stat()
    return {
        "file_name": p.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def canonical_partition(value: Any) -> str:
    text = str(value).strip().lower()
    if text == "valid":
        return "val"
    return text


def load_metadata(data_path: str | Path, *, require_features: bool = False) -> pd.DataFrame:
    cols = list(MANIFEST_COLUMNS)
    if require_features:
        cols.extend(("Reaction", "MappedReaction", "largest_molecule"))
    frame = pd.read_csv(data_path, usecols=cols, dtype={"ID": "string"}, low_memory=False)
    if frame["ID"].isna().any():
        raise ValueError("The ID column contains missing or empty values.")
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].eq("").any():
        raise ValueError("The ID column contains missing or empty values.")
    if frame["ID"].duplicated().any():
        duplicates = frame.loc[frame["ID"].duplicated(), "ID"].head(5).tolist()
        raise ValueError(f"ID values must be unique; duplicates include {duplicates}.")
    frame["Label"] = pd.to_numeric(frame["Label"], errors="raise").astype(np.int8)
    if not set(frame["Label"].unique()).issubset({0, 1}):
        raise ValueError("Label must contain only 0 and 1.")
    for col in SPLIT_COLUMNS:
        frame[col] = frame[col].map(canonical_partition)
        observed = set(frame[col].dropna().unique())
        if not set(PARTITIONS).issubset(observed):
            raise ValueError(
                f"{col} must contain train/val/test; observed={sorted(observed)}"
            )
    return frame


def read_cache_manifest(cache_root: str | Path) -> pd.DataFrame:
    path = Path(cache_root) / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Cache manifest missing: {path}. Run cache_features.py first."
        )
    manifest = pd.read_csv(path, dtype={"ID": "string"}, low_memory=False)
    manifest["ID"] = manifest["ID"].astype(str)
    manifest["Label"] = manifest["Label"].astype(np.int8)
    for col in SPLIT_COLUMNS:
        manifest[col] = manifest[col].map(canonical_partition)
    return manifest


def cache_root_for_run(dry_run: bool, override: str | Path | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_FEATURE_ROOT / ".dry_run" if dry_run else DEFAULT_FEATURE_ROOT


def id_filename(sample_id: str) -> str:
    encoded = quote(str(sample_id), safe="-_.")
    return f"{encoded}.npz"


def sample_cache_path(cache_root: str | Path, feature: str, sample_id: str) -> Path:
    return Path(cache_root) / feature / id_filename(sample_id)


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_save_feature(path: str | Path, vector: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".npz", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, feature=np.asarray(vector, dtype=np.float32))
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_feature_vector(cache_root: str | Path, feature: str, sample_id: str) -> np.ndarray:
    path = sample_cache_path(cache_root, feature, sample_id)
    with np.load(path, allow_pickle=False) as payload:
        vector = payload["feature"].astype(np.float32, copy=False)
    if vector.ndim != 1:
        raise ValueError(f"Expected a 1-D vector in {path}; got {vector.shape}.")
    expected = FEATURE_DIMS[feature]
    if vector.shape[0] != expected:
        raise ValueError(
            f"Feature dimension mismatch for {feature}/{sample_id}: "
            f"expected {expected}, got {vector.shape[0]}"
        )
    return vector


def _load_feature_chunk(
    task: tuple[str, str, list[str]],
) -> sp.csr_matrix:
    cache_root, feature, sample_ids = task
    vectors = [load_feature_vector(cache_root, feature, sample_id) for sample_id in sample_ids]
    return sp.csr_matrix(np.vstack(vectors), dtype=np.float32)


def load_partition_matrix(
    cache_root: str | Path,
    feature: str,
    split_col: str,
    partition: str,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    cache_dir = Path(cache_root) / feature / ".matrix_cache" / split_col
    x_path = cache_dir / f"X_{partition}.npz"
    y_path = cache_dir / f"y_{partition}.npy"
    ids_path = cache_dir / f"ids_{partition}.npy"
    if not (x_path.is_file() and y_path.is_file() and ids_path.is_file()):
        raise FileNotFoundError(
            f"Matrix cache incomplete at {cache_dir}; run build_context_matrices.py first."
        )
    return (
        load_npz(x_path).tocsr().astype(np.float32),
        np.load(y_path, allow_pickle=False).astype(np.int8, copy=False),
        np.load(ids_path, allow_pickle=False).astype(str),
    )


def ensure_matrix_caches(
    cache_root: str | Path,
    manifest: pd.DataFrame,
    *,
    splits: Iterable[str] = SPLIT_COLUMNS,
    features: Iterable[str] = FEATURES,
    chunk_size: int = 2048,
    workers: int = 20,
    executor: ProcessPoolExecutor | None = None,
    force: bool = True,
) -> None:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    cache_root = Path(cache_root)
    cache_manifest_path = cache_root / "manifest.csv"
    manifest_mtime = cache_manifest_path.stat().st_mtime_ns
    tasks = [(split, feature) for split in splits for feature in features]

    def build_task(
        split_col: str,
        feature: str,
        pool: ProcessPoolExecutor,
    ) -> None:
        cache_dir = cache_root / feature / ".matrix_cache" / split_col
        metadata_path = cache_dir / "matrix_meta.json"
        cache_dir.mkdir(parents=True, exist_ok=True)
        part_frames = {
            part: manifest.loc[manifest[split_col] == part, ["ID", "Label"]].reset_index(drop=True)
            for part in PARTITIONS
        }
        expected_meta = {
            "feature": feature,
            "split_col": split_col,
            "manifest_mtime_ns": manifest_mtime,
            "feature_dim": FEATURE_DIMS[feature],
            "partition_rows": {part: len(part_frames[part]) for part in PARTITIONS},
        }
        if not force and metadata_path.is_file():
            try:
                complete = json.loads(metadata_path.read_text(encoding="utf-8")) == expected_meta
                for part in PARTITIONS:
                    complete = complete and all(
                        (cache_dir / f"{name}_{part}{suffix}").is_file()
                        for name, suffix in (("X", ".npz"), ("y", ".npy"), ("ids", ".npy"))
                    )
                if complete:
                    return
            except (OSError, json.JSONDecodeError):
                pass

        # Invalidate the completion marker first. If this build is interrupted, a
        # later run will rebuild all partitions instead of trusting mixed old/new files.
        metadata_path.unlink(missing_ok=True)
        for part, rows in part_frames.items():
            x_path = cache_dir / f"X_{part}.npz"
            y_path = cache_dir / f"y_{part}.npy"
            ids_path = cache_dir / f"ids_{part}.npy"
            all_chunks: list[sp.csr_matrix] = []
            ids = rows["ID"].astype(str).to_numpy()
            labels = rows["Label"].to_numpy(dtype=np.int8)
            pbar = tqdm(
                total=len(rows),
                desc=f"{feature}/{split_col}/{part}",
                unit="rxn",
                position=1,
                leave=False,
                dynamic_ncols=True,
            )
            for start in range(0, len(rows), chunk_size):
                batch_ids = ids[start : start + chunk_size]
                shard_size = max(1, (len(batch_ids) + workers - 1) // workers)
                load_tasks = [
                    (
                        str(cache_root),
                        feature,
                        batch_ids[offset : offset + shard_size].tolist(),
                    )
                    for offset in range(0, len(batch_ids), shard_size)
                ]
                all_chunks.extend(pool.map(_load_feature_chunk, load_tasks))
                pbar.update(len(batch_ids))
            pbar.close()
            matrix = sp.vstack(all_chunks, format="csr", dtype=np.float32) if all_chunks else sp.csr_matrix((0, FEATURE_DIMS[feature]), dtype=np.float32)

            fd, temp_name = tempfile.mkstemp(prefix=f".{x_path.name}.", suffix=".npz", dir=cache_dir)
            os.close(fd)
            try:
                save_npz(temp_name, matrix, compressed=True)
                os.replace(temp_name, x_path)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            for target, array in ((y_path, labels), (ids_path, ids.astype(str))):
                fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".npy", dir=cache_dir)
                os.close(fd)
                try:
                    np.save(temp_name, array, allow_pickle=False)
                    os.replace(temp_name, target)
                finally:
                    if os.path.exists(temp_name):
                        os.unlink(temp_name)
        atomic_json(metadata_path, expected_meta)

    owned_executor = executor is None
    pool = executor or ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    )
    try:
        with tqdm(
            total=len(tasks),
            desc="准备特征矩阵（按 ID 对齐）",
            unit="matrix",
            position=0,
            dynamic_ncols=True,
        ) as task_bar:
            for split_col, feature in tasks:
                build_task(split_col, feature, pool)
                task_bar.update(1)
    finally:
        if owned_executor:
            pool.shutdown(wait=True, cancel_futures=False)


def build_mapped_featurizer(feature_mode: str = MAPPED_FEATURE_MODE) -> Any:
    from src.features_mapped_v2 import MappedV2Featurizer

    return MappedV2Featurizer(feature_mode=feature_mode)


def build_rxnfp_generator(gpu_id: int | None = None) -> Any:
    import torch
    from rxnfp.transformer_fingerprints import (
        RXNBERTFingerprintGenerator,
        get_default_model_and_tokenizer,
    )

    if gpu_id is not None:
        if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
            raise RuntimeError(f"RXNFP requested unavailable CUDA device {gpu_id}.")
        torch.cuda.set_device(gpu_id)
    model, tokenizer = get_default_model_and_tokenizer()
    generator = RXNBERTFingerprintGenerator(model, tokenizer)
    if gpu_id is not None:
        generator.device = torch.device(f"cuda:{gpu_id}")
        generator.model.to(generator.device)
    return generator


def encode_rxnfp_reactions(
    generator: Any,
    reactions: list[str],
    *,
    batch_size: int = 32,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("RXNFP batch_size must be >= 1")
    result = np.zeros((len(reactions), 256), dtype=np.float32)
    for start in range(0, len(reactions), batch_size):
        stop = min(start + batch_size, len(reactions))
        batch = reactions[start:stop]
        try:
            values = np.asarray(generator.convert_batch(batch), dtype=np.float32)
            if values.shape != (len(batch), 256):
                raise ValueError(f"RXNFP returned shape {values.shape}, expected {(len(batch), 256)}")
            result[start:stop] = values
        except Exception:
            for offset, reaction in enumerate(batch):
                try:
                    result[start + offset] = np.asarray(generator.convert(reaction), dtype=np.float32)
                except Exception:
                    result[start + offset] = 0.0
    return result


def encode_base_features(
    frame: pd.DataFrame,
    *,
    want_drfp: bool,
    want_rxnfp: bool,
    want_mapped: bool,
    rxnfp_generator: Any | None,
    mapped_featurizer: Any | None,
    rxnfp_progress: bool = False,
) -> dict[str, np.ndarray | sp.csr_matrix]:
    output: dict[str, np.ndarray | sp.csr_matrix] = {}
    reactions = frame["Reaction"].fillna("").astype(str).tolist()

    if want_drfp:
        from drfp import DrfpEncoder

        try:
            values = DrfpEncoder.encode(
                reactions,
                n_folded_length=N_BITS,
                radius=3,
                rings=True,
            )
            output["drfp"] = sp.csr_matrix(np.asarray(values, dtype=np.float32))
        except Exception:
            per_row = []
            for reaction in reactions:
                try:
                    value = DrfpEncoder.encode(
                        [reaction],
                        n_folded_length=N_BITS,
                        radius=3,
                        rings=True,
                    )[0]
                    per_row.append(np.asarray(value, dtype=np.float32))
                except Exception:
                    per_row.append(np.zeros(N_BITS, dtype=np.float32))
            output["drfp"] = sp.csr_matrix(np.vstack(per_row), dtype=np.float32)

    if want_mapped:
        assert mapped_featurizer is not None
        try:
            output["mapped_v2"] = mapped_featurizer.transform(frame).astype(np.float32)
        except Exception:
            per_row = []
            for i in range(len(frame)):
                try:
                    row = mapped_featurizer.transform(frame.iloc[[i]])[0]
                except Exception:
                    row = np.zeros(FEATURE_DIMS["mapped_v2"], dtype=np.float32)
                per_row.append(np.asarray(row, dtype=np.float32))
            output["mapped_v2"] = np.vstack(per_row).astype(np.float32)

    if want_rxnfp:
        assert rxnfp_generator is not None
        if rxnfp_progress:
            row_bar = tqdm(total=len(reactions), desc="RXNFP 编码", unit="rxn", position=1, leave=False)
        else:
            row_bar = None
        result = encode_rxnfp_reactions(rxnfp_generator, reactions)
        if row_bar is not None:
            row_bar.update(len(reactions))
            row_bar.close()
        output["rxnfp"] = result

    if "drfp" in output and "mapped_v2" in output:
        output["drfp_mapped_v2"] = hstack(
            [output["drfp"], sp.csr_matrix(output["mapped_v2"], dtype=np.float32)],
            format="csr",
        )
    if "rxnfp" in output and "mapped_v2" in output:
        output["rxnfp_mapped_v2"] = hstack(
            [sp.csr_matrix(output["rxnfp"], dtype=np.float32), sp.csr_matrix(output["mapped_v2"], dtype=np.float32)],
            format="csr",
        )
    return output


def vector_at(matrix: np.ndarray | sp.spmatrix, index: int) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.getrow(index).toarray().ravel().astype(np.float32, copy=False)
    return np.asarray(matrix[index], dtype=np.float32).ravel()


