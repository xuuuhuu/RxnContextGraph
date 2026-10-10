from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import queue as queue_module
import tempfile
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
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
    build_model,
    choose_threshold,
    evaluate_predictions,
    feature_root_for_run,
    partition_frame,
    predict,
    read_manifest,
    results_root_for_run,
    set_seed,
    validate_gpu_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train selected edge-v4 context models with a live multi-GPU queue."
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--min-validation-precision", type=float, default=0.90)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.compile(dynamic=True); architecture and hyperparameters are unchanged.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _train_one(arguments: tuple[Any, ...], events: Any) -> dict[str, Any]:
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
        epochs,
        patience,
        min_delta,
        learning_rate,
        hidden_dim,
        num_layers,
        dropout,
        min_validation_precision,
        loader_workers,
        amp,
        compile_model,
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
    set_seed(seed)

    manifest = read_manifest(feature_root)
    train_frame = partition_frame(manifest, split, "train")
    validation_frame = partition_frame(manifest, split, "val")
    train_dataset = CachedEdgeV4Dataset(
        feature_root,
        context_feature_root,
        model_name,
        train_frame,
        split,
        "train",
        packed_graph_root=packed_graph_root,
    )
    validation_dataset = CachedEdgeV4Dataset(
        feature_root,
        context_feature_root,
        model_name,
        validation_frame,
        split,
        "val",
        packed_graph_root=packed_graph_root,
    )
    sample = train_dataset[0]
    node_dim, edge_dim = int(sample.x.shape[1]), int(sample.edge_attr.shape[1])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=loader_workers,
        pin_memory=True,
        persistent_workers=loader_workers > 0,
        **({"prefetch_factor": 2} if loader_workers > 0 else {}),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        persistent_workers=loader_workers > 0,
        **({"prefetch_factor": 2} if loader_workers > 0 else {}),
    )
    base_model = build_model(
        model_name,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model = torch.compile(base_model, dynamic=True) if compile_model else base_model
    positive = int((train_frame["Label"] == 1).sum())
    negative = int((train_frame["Label"] == 0).sum())
    positive_weight = negative / max(positive, 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_state: dict[str, torch.Tensor] | None = None
    best_threshold: dict[str, Any] | None = None
    best_mcc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    actual_epochs = 0
    for epoch in range(1, epochs + 1):
        actual_epochs = epoch
        model.train()
        total_loss = 0.0
        total_samples = 0
        train_batch_total = len(train_loader)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(batch)
                loss = criterion(logits, batch.y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * int(batch.num_graphs)
            total_samples += int(batch.num_graphs)
            if (
                batch_index % progress_every_batches == 0
                or batch_index == train_batch_total
            ):
                events.put(
                    (
                        "batch",
                        key,
                        gpu_id,
                        "train",
                        epoch,
                        batch_index,
                        train_batch_total,
                        float(loss.detach().item()),
                    )
                )

        def report_validation_batch(batch_index: int, batch_total: int) -> None:
            if (
                batch_index % progress_every_batches == 0
                or batch_index == batch_total
            ):
                events.put(
                    (
                        "batch",
                        key,
                        gpu_id,
                        "val",
                        epoch,
                        batch_index,
                        batch_total,
                        None,
                    )
                )

        y_val, probability_val, _ = predict(
            model,
            validation_loader,
            device,
            progress_callback=report_validation_batch,
        )
        threshold = choose_threshold(
            y_val.astype(np.int8), probability_val, min_validation_precision
        )
        metrics = evaluate_predictions(y_val.astype(np.int8), probability_val, threshold["threshold"])
        mcc = float(metrics["mcc"])
        mean_loss = total_loss / max(total_samples, 1)
        events.put(("epoch", key, gpu_id, epoch, mean_loss, mcc))
        if mcc > best_mcc + min_delta:
            best_mcc = mcc
            best_epoch = epoch
            best_threshold = threshold
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in copy.deepcopy(base_model.state_dict()).items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None or best_threshold is None:
        raise RuntimeError(f"Training produced no checkpoint: {key}")
    destination = artifact_dir(results_root, split, model_name, seed)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "model": model_name,
        "split": split,
        "seed": int(seed),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "best_epoch": best_epoch,
        "best_validation_mcc": best_mcc,
        "threshold": best_threshold,
    }
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint.", suffix=".pt", dir=destination)
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination / "checkpoint.pt")
    finally:
        Path(temporary).unlink(missing_ok=True)
    atomic_json(destination / "threshold.json", best_threshold)
    atomic_json(
        destination / "training.json",
        {
            "split": split,
            "model": model_name,
            "seed": int(seed),
            "gpu_id": int(gpu_id),
            "n_train": len(train_dataset),
            "n_validation": len(validation_dataset),
            "positive_weight": positive_weight,
            "actual_epochs": actual_epochs,
            "best_epoch": best_epoch,
            "best_validation_mcc": best_mcc,
            "cpu_threads_per_worker": cpu_threads_per_worker,
            "torch_compile": bool(compile_model),
        },
    )
    del model
    torch.cuda.empty_cache()
    events.put(("finished", key, gpu_id, actual_epochs))
    return {"key": key, "actual_epochs": actual_epochs}


def _drain_events(
    events: Any,
    epoch_bar: tqdm,
    batch_bar: tqdm,
    active: dict[str, int],
    observed_batches: dict[tuple[str, str, int], int],
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
            _, key, gpu, stage, epoch, batch_index, batch_total, loss = event
            progress_key = (key, stage, epoch)
            previous = observed_batches.get(progress_key, 0)
            if batch_index > previous:
                batch_bar.update(batch_index - previous)
                observed_batches[progress_key] = batch_index
            postfix: dict[str, Any] = {
                "job": key,
                "gpu": gpu,
                "stage": stage,
                "epoch": epoch,
                "batch": f"{batch_index}/{batch_total}",
            }
            if loss is not None:
                postfix["loss"] = f"{loss:.4f}"
            batch_bar.set_postfix(**postfix, refresh=False)
        elif event[0] == "epoch":
            _, key, gpu, epoch, loss, mcc = event
            epoch_bar.update(1)
            epoch_bar.set_postfix(
                job=key, gpu=gpu, epoch=epoch, loss=f"{loss:.4f}", val_mcc=f"{mcc:.4f}", refresh=False
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
        args.epochs,
        args.patience,
    ) < 1:
        raise ValueError("Worker, thread, progress, batch, epoch and patience values must be positive.")
    thread_count = str(args.cpu_threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count
    gpus = validate_gpu_ids(args.gpus)
    feature_root = feature_root_for_run(args.dry_run, args.feature_root)
    results_root = results_root_for_run(args.dry_run, args.results_root)
    read_manifest(str(feature_root))
    jobs = [
        (split, model, int(seed))
        for split in args.splits
        for model in args.models
        for seed in args.seeds
    ]
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
                args.epochs,
                args.patience,
                args.min_delta,
                args.learning_rate,
                args.hidden_dim,
                args.num_layers,
                args.dropout,
                args.min_validation_precision,
                args.loader_workers,
                args.amp,
                args.compile_model,
                args.cpu_threads_per_worker,
                args.progress_every_batches,
            )
        )

    context = mp.get_context("spawn")
    errors: list[str] = []
    active: dict[str, int] = {}
    observed_batches: dict[tuple[str, str, int], int] = {}
    with context.Manager() as manager:
        events = manager.Queue()
        pools = {
            gpu: ProcessPoolExecutor(max_workers=args.workers_per_gpu, mp_context=context)
            for gpu in gpus
        }
        futures: dict[Future[Any], int] = {}

        def submit_next(gpu: int) -> None:
            if queues[gpu]:
                futures[pools[gpu].submit(_train_one, queues[gpu].popleft(), events)] = gpu

        for gpu in gpus:
            for _ in range(min(args.workers_per_gpu, len(queues[gpu]))):
                submit_next(gpu)
        model_bar = tqdm(total=len(jobs), desc="Training models", unit="model", dynamic_ncols=True)
        epoch_bar = tqdm(desc="Training epochs", unit="epoch", dynamic_ncols=True)
        batch_bar = tqdm(desc="Processed batches", unit="batch", dynamic_ncols=True)
        try:
            while futures:
                _drain_events(
                    events, epoch_bar, batch_bar, active, observed_batches
                )
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
            _drain_events(events, epoch_bar, batch_bar, active, observed_batches)
        finally:
            model_bar.close()
            epoch_bar.close()
            batch_bar.close()
            for pool in pools.values():
                pool.shutdown(wait=True, cancel_futures=False)
    if errors:
        raise RuntimeError(
            f"Training completed with {len(errors)} failed job(s). First failures:\n"
            + "\n\n".join(errors[:5])
        )
    print(f"Checkpoints saved under: {results_root}")


if __name__ == "__main__":
    main()
