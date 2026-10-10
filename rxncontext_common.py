from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR
DEFAULT_DATA_PATH = PROJECT_DIR / "new_data" / "New_RxnContextGraph.csv"
DEFAULT_FEATURE_ROOT = BASE_DIR / "feature"
DEFAULT_RESULTS_ROOT = BASE_DIR / "results"
DEFAULT_CONTEXT_FEATURE_ROOT = PROJECT_DIR / "context_feature"
# Always import the model implementation included in this package.
sys.path.insert(0, str(PROJECT_DIR))

from src.graph.gat_virtual_context import GATVirtualContextClassifier  # noqa: E402


SPLIT_COLUMNS = (
    "random_split",
    "scaffold_split",
    "reaction_core_split",
    "drfp_kmeans100_greedy_split",
)
PARTITIONS = ("train", "val", "test")
MODELS = (
    "gat_edge_v4_pure",
    "gat_edge_v4_rxnfp",
    "gat_edge_v4_mapped_v2",
    "gat_edge_v4",
    "gat_edge_v4_drfp",
    "gat_edge_v4_drfp_mapped_v2",
)
SEEDS = (42, 43, 44, 45, 46)
CONTEXT_MODES = {
    "gat_edge_v4_pure": "none",
    "gat_edge_v4_rxnfp": "rxnfp",
    "gat_edge_v4_mapped_v2": "mapped",
    "gat_edge_v4": "rxnfp_mapped",
    # Reuse the exact RXNFP virtual-context paths with DRFP inputs.
    "gat_edge_v4_drfp": "rxnfp",
    "gat_edge_v4_drfp_mapped_v2": "rxnfp_mapped",
}
CONTEXT_DIMS = {
    "gat_edge_v4_pure": 0,
    "gat_edge_v4_rxnfp": 256,
    "gat_edge_v4_mapped_v2": 6173,
    "gat_edge_v4": 6429,
    "gat_edge_v4_drfp": 2048,
    "gat_edge_v4_drfp_mapped_v2": 8221,
}
CONTEXT_FEATURE_NAMES = {
    "gat_edge_v4_pure": None,
    "gat_edge_v4_rxnfp": "rxnfp",
    "gat_edge_v4_mapped_v2": "mapped_v2",
    "gat_edge_v4": "rxnfp_mapped_v2",
    "gat_edge_v4_drfp": "drfp",
    "gat_edge_v4_drfp_mapped_v2": "drfp_mapped_v2",
}
MANIFEST_COLUMNS = ("ID", "Label", *SPLIT_COLUMNS)


def canonical_partition(value: Any) -> str:
    text = str(value).strip().lower()
    return "val" if text == "valid" else text


def encoded_id(sample_id: str) -> str:
    encoded = quote(str(sample_id), safe="-_.")
    if not encoded or len(encoded.encode("utf-8")) > 220:
        raise ValueError(f"ID cannot be represented safely as a filename: {sample_id!r}")
    return encoded


def graph_path(feature_root: str | Path, sample_id: str) -> Path:
    return Path(feature_root) / f"{encoded_id(sample_id)}.pt"


def feature_root_for_run(dry_run: bool, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    return DEFAULT_FEATURE_ROOT / ".dry_run" if dry_run else DEFAULT_FEATURE_ROOT


def results_root_for_run(dry_run: bool, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    return DEFAULT_RESULTS_ROOT / ".dry_run" if dry_run else DEFAULT_RESULTS_ROOT


def artifact_dir(results_root: str | Path, split: str, model: str, seed: int) -> Path:
    return Path(results_root) / split / model / f"seed_{seed}"


def model_result_dir(results_root: str | Path, split: str, model: str) -> Path:
    return Path(results_root) / split / model


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_numpy(path: str | Path, value: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".npy", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_torch(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".pt", dir=target.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_source_data(data_path: str | Path, require_reactions: bool = False) -> pd.DataFrame:
    columns = list(MANIFEST_COLUMNS)
    if require_reactions:
        columns.extend(("Reaction", "MappedReaction", "largest_molecule"))
    frame = pd.read_csv(data_path, usecols=columns, dtype={"ID": "string"}, low_memory=False)
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].isna().any() or frame["ID"].eq("").any():
        raise ValueError("ID contains missing or empty values.")
    if frame["ID"].duplicated().any():
        examples = frame.loc[frame["ID"].duplicated(), "ID"].head(5).tolist()
        raise ValueError(f"ID must be unique; duplicates include {examples}.")
    frame["Label"] = pd.to_numeric(frame["Label"], errors="raise").astype(np.int8)
    if not set(frame["Label"].unique()).issubset({0, 1}):
        raise ValueError("Label must contain only 0 and 1.")
    for column in SPLIT_COLUMNS:
        frame[column] = frame[column].map(canonical_partition)
        observed = set(frame[column].dropna().unique())
        if not set(PARTITIONS).issubset(observed):
            raise ValueError(f"{column} lacks train/val/test; observed={sorted(observed)}")
    return frame


def dry_run_subset(
    frame: pd.DataFrame,
    splits: Iterable[str],
    per_partition_class: int,
) -> pd.DataFrame:
    if per_partition_class < 1:
        raise ValueError("--dry-samples-per-partition-class must be >= 1")
    selected: set[int] = set()
    for split in splits:
        for partition in PARTITIONS:
            for label in (0, 1):
                indexes = frame.index[(frame[split] == partition) & (frame["Label"] == label)]
                selected.update(indexes[:per_partition_class].tolist())
    return frame.loc[sorted(selected)].reset_index(drop=True)


def write_manifest(feature_root: str | Path, frame: pd.DataFrame) -> None:
    root = Path(feature_root)
    root.mkdir(parents=True, exist_ok=True)
    frame.loc[:, MANIFEST_COLUMNS].to_csv(root / "manifest.csv", index=False)


@lru_cache(maxsize=8)
def read_manifest(feature_root: str) -> pd.DataFrame:
    path = Path(feature_root) / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Cache manifest missing: {path}. Run cache_features.py first.")
    frame = pd.read_csv(path, dtype={"ID": "string"}, low_memory=False)
    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Cache manifest is missing columns: {sorted(missing)}")
    frame["ID"] = frame["ID"].astype(str)
    frame["Label"] = frame["Label"].astype(np.int8)
    for column in SPLIT_COLUMNS:
        frame[column] = frame[column].map(canonical_partition)
    return frame


def partition_frame(manifest: pd.DataFrame, split: str, partition: str) -> pd.DataFrame:
    if split not in SPLIT_COLUMNS:
        raise ValueError(f"Unknown split: {split}")
    mask = manifest[split] == partition
    result = manifest.loc[mask, ["ID", "Label"]].copy()
    # Keep the row in manifest.csv so a partition can address the packed graph arrays.
    result["_cache_index"] = np.flatnonzero(mask.to_numpy())
    result.reset_index(drop=True, inplace=True)
    if result.empty:
        raise ValueError(f"No rows found for {split}/{partition}.")
    return result


class CachedEdgeV4Dataset(Dataset):
    def __init__(
        self,
        feature_root: str | Path,
        context_feature_root: str | Path,
        model: str,
        frame: pd.DataFrame,
        split: str,
        partition: str,
        packed_graph_root: str | Path | None = None,
    ):
        self.feature_root = Path(feature_root)
        self.context_feature_root = Path(context_feature_root)
        self.model = model
        self.ids = frame["ID"].astype(str).tolist()
        self.labels = frame["Label"].astype(np.float32).to_numpy()
        self.cache_indices = (
            frame["_cache_index"].to_numpy(dtype=np.int64, copy=True)
            if "_cache_index" in frame.columns
            else None
        )
        self.packed_graphs = None
        if packed_graph_root is not None:
            if self.cache_indices is None:
                raise ValueError("Packed graph reads require frame['_cache_index'].")
            self.packed_graphs = PackedEdgeV4Store(packed_graph_root)
            if int(self.cache_indices.max(initial=-1)) >= len(self.packed_graphs):
                raise ValueError(
                    f"Packed graph cache has {len(self.packed_graphs):,} rows but this "
                    f"dataset requests row {int(self.cache_indices.max()):,}."
                )
        self.context_matrix: sp.csr_matrix | None = None
        if CONTEXT_DIMS[self.model] > 0:
            feature_name = CONTEXT_FEATURE_NAMES[self.model]
            if feature_name is None:
                raise RuntimeError(f"Missing context feature mapping for {self.model}")
            matrix, matrix_ids = load_context_partition(
                str(self.context_feature_root), feature_name, split, partition
            )
            expected_ids = np.asarray(self.ids, dtype=str)
            if matrix.shape[0] != len(expected_ids) or not np.array_equal(
                matrix_ids, expected_ids
            ):
                raise ValueError(
                    "Context matrix row order does not match graph manifest for "
                    f"{feature_name}/{split}/{partition}."
                )
            expected_dim = CONTEXT_DIMS[self.model]
            if matrix.shape[1] != expected_dim:
                raise ValueError(
                    f"Context matrix dimension mismatch for {feature_name}: "
                    f"expected {expected_dim}, got {matrix.shape[1]}"
                )
            self.context_matrix = matrix

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        sample_id = self.ids[index]
        if self.packed_graphs is not None:
            data = self.packed_graphs.get(int(self.cache_indices[index]))
        else:
            path = graph_path(self.feature_root, sample_id)
            if not path.is_file():
                raise FileNotFoundError(f"Missing graph cache: {path}")
            data = torch.load(path, map_location="cpu", weights_only=False)
        data.y = torch.tensor([self.labels[index]], dtype=torch.float32)
        if self.context_matrix is not None:
            context = self.context_matrix.getrow(index).toarray().reshape(-1).astype(
                np.float32, copy=False
            )
            if context.shape != (CONTEXT_DIMS[self.model],):
                raise ValueError(
                    f"Context shape mismatch for {sample_id}: expected "
                    f"{(CONTEXT_DIMS[self.model],)}, got {context.shape}"
                )
            data.mapped_x = torch.from_numpy(context).view(1, -1)
        return data


class PackedEdgeV4Store:
    """Read edge-v4 model inputs from contiguous, memory-mapped arrays."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Packed graph metadata missing: {metadata_path}. "
                "Run pack_graph_cache.py first."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format") != "edge_v4_contiguous_v1":
            raise ValueError(f"Unsupported packed graph format in {metadata_path}")
        self.count = int(self.metadata["count"])
        self.node_dim = int(self.metadata["node_dim"])
        self.edge_dim = int(self.metadata["edge_dim"])
        self.node_offsets = np.load(self.root / "node_offsets.npy", mmap_mode="r")
        self.edge_offsets = np.load(self.root / "edge_offsets.npy", mmap_mode="r")
        if self.node_offsets.shape != (self.count + 1,) or self.edge_offsets.shape != (
            self.count + 1,
        ):
            raise ValueError(f"Invalid packed graph offsets under {self.root}")
        total_nodes = int(self.node_offsets[-1])
        total_edges = int(self.edge_offsets[-1])
        # mode='c' gives writable private views accepted by torch.from_numpy while
        # retaining file-backed, copy-on-write storage shared by worker processes.
        self.x = np.memmap(
            self.root / "x.f32", dtype=np.float32, mode="c", shape=(total_nodes, self.node_dim)
        )
        self.edge_attr = np.memmap(
            self.root / "edge_attr.f32",
            dtype=np.float32,
            mode="c",
            shape=(total_edges, self.edge_dim),
        )
        self.edge_index = np.memmap(
            self.root / "edge_index.i64", dtype=np.int64, mode="c", shape=(total_edges, 2)
        )
        self.center_mask = np.memmap(
            self.root / "center_mask.f32", dtype=np.float32, mode="c", shape=(total_nodes,)
        )
        self.center_local_mask = np.memmap(
            self.root / "center_local_mask.f32",
            dtype=np.float32,
            mode="c",
            shape=(total_nodes,),
        )

    def __len__(self) -> int:
        return self.count

    def get(self, index: int) -> Data:
        node_start, node_end = map(int, self.node_offsets[index : index + 2])
        edge_start, edge_end = map(int, self.edge_offsets[index : index + 2])
        return Data(
            x=torch.from_numpy(self.x[node_start:node_end]),
            edge_index=torch.from_numpy(self.edge_index[edge_start:edge_end]).t(),
            edge_attr=torch.from_numpy(self.edge_attr[edge_start:edge_end]),
            center_mask=torch.from_numpy(self.center_mask[node_start:node_end]),
            center_local_mask=torch.from_numpy(
                self.center_local_mask[node_start:node_end]
            ),
        )


@lru_cache(maxsize=2)
def load_context_partition(
    context_feature_root: str,
    feature_name: str,
    split: str,
    partition: str,
) -> tuple[sp.csr_matrix, np.ndarray]:
    cache_dir = Path(context_feature_root) / feature_name / ".matrix_cache" / split
    matrix_path = cache_dir / f"X_{partition}.npz"
    ids_path = cache_dir / f"ids_{partition}.npy"
    if not matrix_path.is_file() or not ids_path.is_file():
        raise FileNotFoundError(
            f"Context matrix cache is incomplete at {cache_dir}; expected "
            f"{matrix_path.name} and {ids_path.name}."
        )
    matrix = sp.load_npz(matrix_path).tocsr().astype(np.float32)
    ids = np.load(ids_path, allow_pickle=False).astype(str)
    return matrix, ids


def build_model(
    model_name: str,
    *,
    node_dim: int,
    edge_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> torch.nn.Module:
    return GATVirtualContextClassifier(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        mapped_dim=CONTEXT_DIMS[model_name],
        context_mode=CONTEXT_MODES[model_name],
        dropout=dropout,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_gpu_ids(gpus: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(gpu) for gpu in gpus)
    if not values:
        raise ValueError("At least one GPU ID is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in ml_env.")
    count = torch.cuda.device_count()
    invalid = [gpu for gpu in values if gpu < 0 or gpu >= count]
    if invalid:
        raise ValueError(f"Unavailable GPU IDs {invalid}; visible device count={count}.")
    return values


def choose_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    min_precision: float = 0.90,
) -> dict[str, Any]:
    rows: list[tuple[float, float, float]] = []
    valid: list[tuple[float, float, float]] = []
    for threshold in np.arange(0.10, 0.9500001, 0.01):
        prediction = (y_probability >= threshold).astype(np.int8)
        precision = float(precision_score(y_true, prediction, zero_division=0))
        recall = float(recall_score(y_true, prediction, zero_division=0))
        row = (float(threshold), precision, recall)
        rows.append(row)
        if precision >= min_precision:
            valid.append(row)
    pool = valid if valid else rows
    best = max(pool, key=lambda row: (row[2] if valid else row[1], row[1] if valid else row[2], -row[0]))
    return {
        "threshold": best[0],
        "precision": best[1],
        "recall": best[2],
        "constraint_satisfied": bool(valid),
        "min_precision": float(min_precision),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    prediction = (y_probability >= threshold).astype(np.int8)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    return {
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "f0_5": float(fbeta_score(y_true, prediction, beta=0.5, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "ap": float(average_precision_score(y_true, y_probability)),
        "confusion_matrix": matrix.tolist(),
        "threshold": float(threshold),
    }


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
):
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    ids: list[str] = []
    with torch.inference_mode():
        offset = 0
        total_batches = len(loader)
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=True)
            probabilities.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
            labels.append(batch.y.view(-1).detach().cpu().numpy())
            count = int(batch.num_graphs)
            if hasattr(loader.dataset, "ids"):
                ids.extend(loader.dataset.ids[offset : offset + count])
            offset += count
            if progress_callback is not None:
                progress_callback(batch_index, total_batches)
    return np.concatenate(labels), np.concatenate(probabilities), ids


def batch_count(size: int, batch_size: int) -> int:
    return int(math.ceil(size / batch_size))
