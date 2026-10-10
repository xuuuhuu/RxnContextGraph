from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_module
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from rxncontext_common import (
    MODELS,
    SEEDS,
    SPLIT_COLUMNS,
    DEFAULT_CONTEXT_FEATURE_ROOT,
    CachedEdgeV4Dataset,
    artifact_dir,
    atomic_json,
    batch_count,
    build_model,
    evaluate_predictions,
    feature_root_for_run,
    model_result_dir,
    partition_frame,
    read_manifest,
    results_root_for_run,
    validate_gpu_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test selected edge-v4 context checkpoints and aggregate five seeds."
    )
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument(
        "--packed-graph-root",
        type=Path,
        default=None,
        help="Contiguous graph cache created by pack_graph_cache.py.",
    )
    parser.add_argument(
        "--context-feature-root",
        type=Path,
        default=DEFAULT_CONTEXT_FEATURE_ROOT,
        help="Existing feature+machine_learning/feature cache root.",
    )
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=SPLIT_COLUMNS, default=list(SPLIT_COLUMNS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=4)
    parser.add_argument("--progress-every-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _test_one(arguments: tuple[Any, ...], events: Any) -> dict[str, Any]:
    (
        split,
        model_name,
        seed,
        gpu_id,
        feature_root,
        packed_graph_root,
        context_feature_root,
        results_root,
        batch_size,
        loader_workers,
        cpu_threads_per_worker,
        progress_every_batches,
    ) = arguments
    key = f"{split}/{model_name}/seed_{seed}"
    events.put(("started", key, gpu_id))
    torch.set_num_threads(cpu_threads_per_worker)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    destination = artifact_dir(results_root, split, model_name, seed)
    checkpoint_path = destination / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(
        model_name,
        node_dim=int(checkpoint["node_dim"]),
        edge_dim=int(checkpoint["edge_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    manifest = read_manifest(feature_root)
    test_frame = partition_frame(manifest, split, "test")
    dataset = CachedEdgeV4Dataset(
        feature_root,
        context_feature_root,
        model_name,
        test_frame,
        split,
        "test",
        packed_graph_root=packed_graph_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        persistent_workers=loader_workers > 0,
        **({"prefetch_factor": 2} if loader_workers > 0 else {}),
    )
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    ids: list[str] = []
    offset = 0
    with torch.inference_mode():
        batch_total = len(loader)
        for batch_index, batch in enumerate(loader, start=1):
            count = int(batch.num_graphs)
            ids.extend(dataset.ids[offset : offset + count])
            offset += count
            batch = batch.to(device, non_blocking=True)
            probability = torch.sigmoid(model(batch)).detach().cpu().numpy()
            labels.append(batch.y.view(-1).detach().cpu().numpy())
            probabilities.append(probability)
            if (
                batch_index % progress_every_batches == 0
                or batch_index == batch_total
            ):
                events.put(("batch", key, gpu_id, batch_index, batch_total))
    y_true = np.concatenate(labels).astype(np.int8)
    y_probability = np.concatenate(probabilities)
    threshold = float(checkpoint["threshold"]["threshold"])
    metrics = evaluate_predictions(y_true, y_probability, threshold)
    metrics.update(
        {
            "split": split,
            "model": model_name,
            "seed": int(seed),
            "threshold_source": "validation",
            "best_epoch": int(checkpoint["best_epoch"]),
        }
    )
    prediction = (y_probability >= threshold).astype(np.int8)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ID": ids,
            "y_true": y_true,
            "y_probability": y_probability,
            "threshold": threshold,
            "y_pred": prediction,
        }
    ).to_csv(destination / "test_predictions.csv", index=False)
    atomic_json(destination / "metrics.json", metrics)
    del model
    torch.cuda.empty_cache()
    events.put(("finished", key, gpu_id))
    return metrics


def _write_aggregate(results_root: Path, split: str, model_name: str, seeds: list[int]) -> None:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        path = artifact_dir(results_root, split, model_name, seed) / "metrics.json"
        if path.is_file():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        return
    destination = model_result_dir(results_root, split, model_name)
    seed_frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    seed_frame["confusion_matrix"] = seed_frame["confusion_matrix"].map(
        lambda value: json.dumps(value, separators=(",", ":"))
    )
    seed_frame.to_csv(destination / "seed_results.csv", index=False)

    raw = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "n_seeds": len(rows),
        "seeds": ",".join(str(seed) for seed in sorted(raw["seed"].astype(int).tolist())),
    }
    for metric in ("mcc", "f0_5", "precision", "recall", "average_precision", "ap"):
        values = pd.to_numeric(raw[metric], errors="raise")
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    matrices = np.asarray(raw["confusion_matrix"].tolist(), dtype=np.int64)
    summary["confusion_matrix_sum"] = json.dumps(
        matrices.sum(axis=0).tolist(), separators=(",", ":")
    )
    summary["confusion_matrix_mean"] = json.dumps(
        matrices.mean(axis=0).tolist(), separators=(",", ":")
    )
    pd.DataFrame([summary]).to_csv(destination / "summary.csv", index=False)


def _drain_events(
    events: Any,
    batch_bar: tqdm,
    active: dict[str, int],
    observed_batches: dict[str, int],
) -> None:
    while True:
        try:
            event = events.get_nowait()
        except queue_module.Empty:
            return
        if event[0] == "started":
            _, key, gpu = event
            active[key] = gpu
            batch_bar.set_postfix(job=key, gpu=gpu, stage="loading", refresh=True)
        elif event[0] == "batch":
            _, key, gpu, batch_index, batch_total = event
            previous = observed_batches.get(key, 0)
            if batch_index > previous:
                batch_bar.update(batch_index - previous)
                observed_batches[key] = batch_index
            batch_bar.set_postfix(
                job=key,
                gpu=gpu,
                batch=f"{batch_index}/{batch_total}",
                refresh=False,
            )
        elif event[0] == "finished":
            active.pop(event[1], None)


def main() -> None:
    args = parse_args()
    if min(
        args.workers_per_gpu,
        args.cpu_threads_per_worker,
        args.progress_every_batches,
        args.batch_size,
    ) < 1:
        raise ValueError("Worker, thread, progress and batch sizes must be positive.")
    thread_count = str(args.cpu_threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count
    gpus = validate_gpu_ids(args.gpus)
    feature_root = feature_root_for_run(args.dry_run, args.feature_root)
    results_root = results_root_for_run(args.dry_run, args.results_root)
    manifest = read_manifest(str(feature_root))
    jobs: list[tuple[str, str, int]] = []
    total_batches = 0
    for split in args.splits:
        test_size = len(partition_frame(manifest, split, "test"))
        for model in args.models:
            for seed in args.seeds:
                jobs.append((split, model, int(seed)))
                total_batches += batch_count(test_size, args.batch_size)

    queues: dict[int, deque[tuple[Any, ...]]] = {gpu: deque() for gpu in gpus}
    for index, (split, model, seed) in enumerate(jobs):
        gpu = gpus[index % len(gpus)]
        queues[gpu].append(
            (
                split,
                model,
                seed,
                gpu,
                str(feature_root),
                str(args.packed_graph_root.expanduser().resolve())
                if args.packed_graph_root is not None
                else None,
                str(args.context_feature_root.expanduser().resolve()),
                str(results_root),
                args.batch_size,
                args.loader_workers,
                args.cpu_threads_per_worker,
                args.progress_every_batches,
            )
        )
    context = mp.get_context("spawn")
    errors: list[str] = []
    active: dict[str, int] = {}
    observed_batches: dict[str, int] = {}
    with context.Manager() as manager:
        events = manager.Queue()
        pools = {
            gpu: ProcessPoolExecutor(max_workers=args.workers_per_gpu, mp_context=context)
            for gpu in gpus
        }
        futures: dict[Future[Any], int] = {}

        def submit_next(gpu: int) -> None:
            if queues[gpu]:
                futures[pools[gpu].submit(_test_one, queues[gpu].popleft(), events)] = gpu

        for gpu in gpus:
            for _ in range(min(args.workers_per_gpu, len(queues[gpu]))):
                submit_next(gpu)
        model_bar = tqdm(total=len(jobs), desc="Testing models", unit="model", dynamic_ncols=True)
        batch_bar = tqdm(total=total_batches, desc="Testing batches", unit="batch", dynamic_ncols=True)
        try:
            while futures:
                _drain_events(events, batch_bar, active, observed_batches)
                done, _ = wait(tuple(futures), timeout=0.25, return_when=FIRST_COMPLETED)
                for future in done:
                    gpu = futures.pop(future)
                    try:
                        future.result()
                    except Exception as error:
                        errors.append(
                            f"GPU {gpu}: {type(error).__name__}: {error}\n{traceback.format_exc()}"
                        )
                    model_bar.update(1)
                    submit_next(gpu)
            _drain_events(events, batch_bar, active, observed_batches)
        finally:
            model_bar.close()
            batch_bar.close()
            for pool in pools.values():
                pool.shutdown(wait=True, cancel_futures=False)
    if errors:
        raise RuntimeError(
            f"Testing completed with {len(errors)} failed job(s). First failures:\n"
            + "\n\n".join(errors[:5])
        )
    for split in args.splits:
        for model in args.models:
            _write_aggregate(results_root, split, model, [int(seed) for seed in args.seeds])
    print(f"Test metrics and five-seed summaries saved under: {results_root}")


if __name__ == "__main__":
    main()
