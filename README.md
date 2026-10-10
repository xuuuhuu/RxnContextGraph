# RxnContextGraph: `gat_edge_v4` reproduction package

This package contains the feature and graph builders, the `gat_edge_v4` model, training and evaluation scripts, and released results for four data splits. It does **not** include the input dataset, graph caches, or context feature caches. Download the dataset and generate both caches before training or running checkpoint inference.

The four splits are `random_split`, `scaffold_split`, `reaction_core_split`, and `drfp_kmeans100_greedy_split`. Each has results for seeds 42–46.

## 1. Released results

**The only released `checkpoint.pt` is for `drfp_kmeans100_greedy_split`, seed 42.** The other 19 runs retain per-seed test metrics, validation-threshold records, training records, and per-sample test predictions, but no checkpoints. To infer with one of those runs, train it again first. The released predictions are sufficient to recompute all released test metrics and split summaries without training.

Each `results/<split>/gat_edge_v4/seed_<seed>/` contains:

| File | Contents |
| --- | --- |
| `metrics.json` | Test-set metrics. `mcc` is Matthews correlation coefficient; `f0_5` is F0.5; `precision` and `recall` use the selected threshold; `average_precision` and `ap` are the same probability-based average precision; `confusion_matrix` is `[[TN, FP], [FN, TP]]` with class order `[0, 1]`; `threshold` was selected on validation data. `split`, `model`, and `seed` identify the run; `threshold_source` is `validation`; `best_epoch` identifies the selected training epoch. |
| `test_predictions.csv` | One row per test sample. `ID` matches the input CSV; `y_true` is the binary label; `y_probability` is the predicted positive-class probability; `threshold` is the seed's validation-selected threshold; `y_pred` equals `y_probability >= threshold`. |
| `threshold.json` | Validation-set threshold selection: `threshold`, its validation `precision` and `recall`, `constraint_satisfied`, and `min_precision`. The script searches 0.10 through 0.95 in steps of 0.01; the default precision floor is 0.90. Among candidates meeting the floor it maximizes recall, then precision, then prefers the lower threshold. If none meet it, it maximizes precision, then recall, then prefers the lower threshold. The test set does not select the threshold, so its precision may be below 0.90. |
| `training.json` | `split`, `model`, `seed`, `gpu_id`, training and validation counts (`n_train`, `n_validation`), `positive_weight` (training negatives divided by positives for weighted BCE), `actual_epochs`, `best_epoch`, `best_validation_mcc`, `cpu_threads_per_worker`, and `torch_compile`. |
| `checkpoint.pt` | Present only for the greedy split, seed 42. It stores the selected epoch's `model_state`, architecture settings, validation threshold, split, and seed. |

Each `results/<split>/gat_edge_v4/` also contains `seed_results.csv` (five per-seed test rows) and `summary.csv` (means and population standard deviations, `ddof=0`). Confusion matrices are reported as five-seed sums and means.

## 2. Dataset download and file layout

Download the processed reaction dataset from [this Google Drive file](https://drive.google.com/file/d/1sMuVMI8opXV3z24dzC-Q-LuMJIPtUF-z/view?usp=drive_link). Save it as **`new_data/New_RxnContextGraph.csv`** under the project root. Create `new_data/` if it is absent. The file must contain `Reaction`, `MappedReaction`, `ID`, `Label`, and the four split columns listed above, each with `train`, `val`, and `test` assignments.

The input CSV used for the released experiments contained 1,135,643 rows and had SHA-256:

```text
3683805e5a6c00e0720c3fb29d989ce28a5cf12f8604e7754ba2bf85c6ccfb8c
```

Compare this digest with the downloaded file before expecting matching results. The download link's anonymous access and file contents could not be independently verified from this environment.

```text
RxnContextGraph/
├── new_data/New_RxnContextGraph.csv      # download separately; not included
├── context_pipeline/
│   ├── cache_features.py                 # RXNFP, mapped_v2, combined features
│   └── ml_common.py                      # encoding and matrix assembly
├── src/features_mapped_v2.py             # mapped-reaction fingerprints/descriptors
├── src/graph/                            # edge-v4 builder, fallbacks, GAT model
├── build_context_matrices.py             # split-specific train/val/test matrices
├── cache_features.py                     # one edge-v4 graph per reaction ID
├── pack_graph_cache.py                   # optional contiguous graph storage
├── rxncontext_common.py                  # data loading, model, threshold, metrics
├── train_models.py                       # training
├── test_models.py                        # checkpoint inference and aggregation
├── reproduce_results.py                 # metrics, tables, derived comparison plot
└── results/<split>/gat_edge_v4/           # released results
```

`reproduction/` is created when you run the commands below. Its generated files are:

| Path | Produced data |
| --- | --- |
| `reproduction/context_feature/rxnfp/` | Per-ID `.npz` vectors: 256-dimensional RXNFP from `Reaction`. |
| `reproduction/context_feature/mapped_v2/` | Per-ID `.npz` vectors: a 2,048-dimensional Morgan difference from `Reaction`, 4,096 reaction-center fingerprint values from `MappedReaction`, and 29 descriptors (6,173 total). |
| `reproduction/context_feature/rxnfp_mapped_v2/` | Per-ID concatenation of the preceding features (6,429 dimensions). Its `.matrix_cache/<split>/` is generated separately and contains `X_train/val/test.npz`, `ids_*.npy`, `y_*.npy`, and `matrix_meta.json`. |
| `reproduction/feature/` | One PyTorch Geometric `.pt` graph per ID, plus `manifest.csv` and `cache_metadata.json`. Graphs store node and edge features, connectivity, reaction-center masks, and the label. If edge-v4 construction fails, the builder tries padded edge-v3, then padded edge-v2, then a dummy graph. |
| `reproduction/results/` | New checkpoints, predictions, and metrics from training and testing. |

Training reads graph IDs, labels, and split assignments from `manifest.csv`, then loads the context matrix in matching ID order. Training and testing do not generate missing graph or context features automatically.

The downloaded CSV is the starting point for this package. Upstream reaction cleaning, atom mapping, and split-generation scripts are not included, so the package cannot recreate that CSV from earlier raw sources. The package does not include a license or a per-record provenance manifest for the dataset or copied model code; confirm redistribution terms before publishing them elsewhere.

## 3. Environment

The original Linux/NVIDIA GPU environment was `ml_env`. RXNFP generation, training, and checkpoint inference need a working CUDA GPU. Recomputing metrics from released predictions does not.

| Component | Checked version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch / PyTorch Geometric | 2.11.0+cu130 / 2.7.0 |
| CUDA / cuDNN | 13.0 / 9.19.0 (`torch.backends.cudnn.version()` returns 91900) |
| RDKit | 2025.09.5 |
| NumPy / pandas / SciPy | 2.4.3 / 3.0.1 / 1.17.1 |
| scikit-learn / matplotlib / tqdm | 1.8.0 / 3.10.8 / 4.67.3 |
| rxnfp / transformers / threadpoolctl | 0.1.0 / 4.30.2 / 3.6.0 |

On Linux x86-64 with support for the CUDA 13.0 PyTorch wheel:

```bash
conda create -n ml_env python=3.11.15 pip -y
conda activate ml_env
python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu130
python -m pip install 'torch-geometric==2.7.0' 'rdkit==2025.9.5' \
  'numpy==2.4.3' 'pandas==3.0.1' 'scipy==1.17.1' \
  'scikit-learn==1.8.0' 'matplotlib==3.10.8' 'tqdm==4.67.3' \
  'rxnfp==0.1.0' 'transformers==4.30.2' 'threadpoolctl==3.6.0'
```

See the [PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/), [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), and [RXNFP project](https://github.com/rxn4chemistry/rxnfp). RXNFP also needs access to its pretrained model on first use.

```bash
python - <<'PYENV'
import torch, torch_geometric, rdkit, rxnfp, transformers
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('PyG:', torch_geometric.__version__, 'RDKit:', rdkit.__version__)
print('rxnfp:', rxnfp.__version__, 'transformers:', transformers.__version__)
print('cuDNN:', torch.backends.cudnn.version(), 'GPU:', torch.cuda.is_available())
PYENV
```

## 4. Run order: downloaded CSV to results

Run these commands from the project root. GPU `0` is an example; replace it with an available device. All generated files go under `reproduction/`, keeping the released `results/` untouched.

```bash
cd /path/to/RxnContextGraph  # replace with the extracted project directory
conda activate ml_env
RXN_ROOT="$PWD"
RXN_RUN="$RXN_ROOT/reproduction"
RXN_CONTEXT_ROOT="$RXN_RUN/context_feature"
RXN_GRAPH_ROOT="$RXN_RUN/feature"
mkdir -p "$RXN_ROOT/new_data" "$RXN_RUN"
# Place the downloaded file at $RXN_ROOT/new_data/New_RxnContextGraph.csv
DATA_FILE="$RXN_ROOT/new_data/New_RxnContextGraph.csv"
test -s "$DATA_FILE"
sha256sum "$DATA_FILE"
```

### 4.1 Generate context features first

The feature script processes the CSV in RXNFP, mapped-v2, then concatenation phases. The second script assembles 6,429-dimensional matrices for train, validation, and test in all four splits. The full dataset contains over 1.1 million reactions; this stage takes time and creates many files, so allow ample disk space.

```bash
python -m context_pipeline.cache_features \
  --data-path "$DATA_FILE" \
  --features rxnfp mapped_v2 rxnfp_mapped_v2 \
  --cache-root "$RXN_CONTEXT_ROOT" \
  --rxnfp-gpus 0 --rxnfp-workers 1 --cpu-workers 20

python build_context_matrices.py \
  --cache-root "$RXN_CONTEXT_ROOT" \
  --workers 20
```

### 4.2 Build edge-v4 reaction graphs

`cache_features.py` reads `ID`, `MappedReaction`, and `Label` from the same CSV. It writes a graph per ID for all splits and seeds and creates the `manifest.csv` used by training.

```bash
python cache_features.py \
  --data-path "$DATA_FILE" \
  --feature-root "$RXN_GRAPH_ROOT" \
  --graph-workers 20
```

Optionally pack the graphs into contiguous arrays. If you do, add `--packed-graph-root "$RXN_RUN/packed_edge_v4"` to **both** the training and testing commands.

```bash
python pack_graph_cache.py \
  --feature-root "$RXN_GRAPH_ROOT" \
  --output-root "$RXN_RUN/packed_edge_v4" \
  --workers 20
```

### 4.3 Train and test four splits × five seeds

Complete steps 4.1 and 4.2 first. The defaults are batch size 32, at most 20 epochs, patience 5, learning rate 0.001, hidden dimension 128, three layers, and dropout 0.2. `--compile` matches the released seed-42 training record. Re-training on different hardware or software may not yield bitwise-identical weights.

```bash
python train_models.py \
  --models gat_edge_v4 \
  --splits random_split scaffold_split reaction_core_split drfp_kmeans100_greedy_split \
  --seeds 42 43 44 45 46 \
  --feature-root "$RXN_GRAPH_ROOT" \
  --context-feature-root "$RXN_CONTEXT_ROOT" \
  --results-root "$RXN_RUN/results" \
  --gpus 0 --workers-per-gpu 1 --compile

python test_models.py \
  --models gat_edge_v4 \
  --splits random_split scaffold_split reaction_core_split drfp_kmeans100_greedy_split \
  --seeds 42 43 44 45 46 \
  --feature-root "$RXN_GRAPH_ROOT" \
  --context-feature-root "$RXN_CONTEXT_ROOT" \
  --results-root "$RXN_RUN/results" \
  --gpus 0 --workers-per-gpu 1
```

Train before testing because `test_models.py` loads each seed's checkpoint. Only one checkpoint is provided in the release.

### 4.4 Recompute metrics, tables, and a derived plot from released predictions

This step can run independently of steps 4.1–4.3. It reads all 20 released `test_predictions.csv` files, uses each run's `threshold.json` to recompute MCC, F0.5, precision, recall, average precision, and the confusion matrix, and checks `metrics.json`. It then summarizes five seeds per split using population standard deviations. No input CSV, GPU, graphs, features, or checkpoint are needed.

```bash
python reproduce_results.py \
  --results-root "$RXN_ROOT/results" \
  --output-dir "$RXN_RUN/derived_public"
```

The outputs are `recomputed_seed_metrics.csv` (20 rows), `recomputed_split_summary.csv` (4 rows), `gat_edge_v4_split_metrics.png`, and an SVG with the same base name. The plot is newly drawn from the released predictions; it is **not a reproduction of an original manuscript figure**. To recompute newly trained results after step 4.3:

```bash
python reproduce_results.py \
  --results-root "$RXN_RUN/results" \
  --output-dir "$RXN_RUN/derived_new"
```

### 4.5 Run inference with the released greedy-split seed-42 checkpoint

Complete steps 4.1 and 4.2 first. Copy the released checkpoint to a separate result directory and test only that run.

```bash
RXN_SEED_DIR="$RXN_RUN/seed42_results/drfp_kmeans100_greedy_split/gat_edge_v4/seed_42"
mkdir -p "$RXN_SEED_DIR"
cp "$RXN_ROOT/results/drfp_kmeans100_greedy_split/gat_edge_v4/seed_42/checkpoint.pt" "$RXN_SEED_DIR/"
python test_models.py \
  --models gat_edge_v4 \
  --splits drfp_kmeans100_greedy_split \
  --seeds 42 \
  --feature-root "$RXN_GRAPH_ROOT" \
  --context-feature-root "$RXN_CONTEXT_ROOT" \
  --results-root "$RXN_RUN/seed42_results" \
  --gpus 0 --workers-per-gpu 1
```
