# MARS-Rec

A simplified, supervised-only descendant of [MARS-Gym](https://github.com/deeplearningbrasil/mars-gym) (formerly *MARS-Gym Lite*) — a framework for modeling, training, and evaluating recommendation systems with deep learning models.

MARS-Rec removes all Reinforcement Learning, bandits, gym environments, and policy evaluation layers from the original project, keeping only the **data preparation** and **supervised training** pipeline for neural models.

## Key differences from MARS-Gym

| Aspect | MARS-Gym | MARS-Rec |
|---|---|---|
| Data processing | PySpark + Pandas | Pandas only |
| Training | Supervised + Bandit/RL | Supervised only |
| Evaluation | Bandit policies, counterfactual | Simple score-based ranking |
| Prediction | BanditAgent with policies (e-greedy, etc.) | Direct ranking by model score |
| Orchestration | Luigi | Luigi (kept) |
| Training loop | Torchbearer | PyTorch Lightning |
| Loss functions | BCE, Focal, CRM (Counterfactual Risk) | BCE, Focal, Implicit BCE, Dummy |
| CLI | run supervised / interaction / evaluate / viz | run supervised / run data / list / results / predict / evaluate |

## Architecture

```
mars-rec/
├── pyproject.toml                    # Package metadata, dependencies, ruff/pytest config
├── requirements.lock                 # Pinned dependency tree (Dockerfile + CI); see lock-constraints.txt
├── Makefile                          # Dev / Docker / Vertex AI targets
├── Dockerfile                        # CPU/GPU image (default base: pytorch/pytorch:2.14.1-cuda12.6)
├── src/
│   └── mars_rec/
│       ├── __init__.py
│       ├── cli.py                    # Click CLI (run supervised, run data, list, results, predict, evaluate, visualize)
│       ├── cuda.py                   # CUDA device management
│       ├── filtering.py              # CandidateProvider / CandidateFilter (inference-time candidate pools)
│       ├── inference.py              # Standalone inference + evaluation (run_inference, run_evaluate)
│       ├── meta_config.py            # ProjectConfig, Column, IOType, RecommenderType
│       ├── luigi.cfg                 # Default Luigi configuration
│       ├── visualization.py          # Architecture diagram (SVG) + text summary
│       ├── data/
│       │   ├── dataset.py            # InteractionsDataset, negative sampling datasets
│       │   ├── task.py               # BasePrepareDataFrames (split, balancing, etc.)
│       │   └── utils.py              # DownloadDataset, load_dataset
│       ├── model/
│       │   ├── abstract.py           # RecommenderModule (base class for all models)
│       │   └── base_model.py         # LogisticRegression (baseline model)
│       ├── simulation/
│       │   └── training.py           # SupervisedModelTraining + run_evaluate (full pipeline)
│       ├── torch/
│       │   ├── callbacks.py          # Lightning callbacks
│       │   ├── data.py               # FasterBatchSampler, NoAutoCollationDataLoader
│       │   ├── init.py               # Weight initializers (lecun, he)
│       │   ├── lightning_module.py   # RecommenderLightningModule (Lightning wrapper)
│       │   ├── loss.py               # Focal, ImplicitFeedbackBCE, BPR, InfoNCE, RankNet, LambdaRank, ...
│       │   ├── metrics.py            # BCE, binary_accuracy, precision, recall, f1, mse, mae, rmse
│       │   ├── optimizer.py          # RAdam
│       │   └── summary.py            # Model summary (Keras-style)
│       └── utils/
│           ├── files.py              # Output paths (weights, history, params, etc.)
│           ├── index_mapping.py      # Categorical → integer index mapping
│           ├── logging.py            # Logger setup
│           ├── plot.py               # plot_history (training curves)
│           ├── reflection.py         # load_attr (dynamic class import)
│           └── utils.py              # parallel_literal_eval, reduce_df_mem
├── tests/                            # Pytest suite
└── examples/                         # Reference recsys projects (public datasets, auto-download)
    ├── movielens/                    # MovieLens 1M — NCF (collaborative filtering with side features)
    ├── lastfm/                       # Last.fm HetRec 2K — sequential SASRec-lite
    ├── online_retail/                # UCI Online Retail — session-based GRU4Rec (session_id as user)
    ├── amazon_beauty/                # Amazon Beauty 5-core — content-based two-tower (text emb)
    └── movielens100k_slate/          # MovieLens 100K — slate ranking (available_items + metadata)
        # Each subdir contains: config.py, data.py, model.py, run_training.py
```

## Installation

```bash
git clone git@github.com:marlesson/mars-rec.git
cd mars-rec

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .
```

### Main dependencies

- `torch >= 2.4` — PyTorch for deep learning models
- `pytorch-lightning >= 2.4, < 3` — Training loop with callbacks and logging
- `luigi >= 3.5, < 4` — Data and training pipeline orchestration
- `pandas >= 2.2` — Data manipulation (Pandas 3.x is supported and used in CI)
- `scikit-learn >= 1.5` — Data splitting, metrics
- `imbalanced-learn >= 0.13` — Over/undersampling for class balancing

Python 3.11–3.13 is supported.

For reproducible builds, the full pinned dependency tree lives in
`requirements.lock`. It is resolved for Python 3.11 (the oldest supported
version), so it installs on every supported version. Lock-only pins, such as
keeping torch on a CUDA 12.x release, live in `lock-constraints.txt`.

```bash
make lock          # re-resolves requirements.lock after editing pyproject.toml
make lock-upgrade  # bumps every package to the newest allowed version
make audit         # checks the lockfile against the OSV vulnerability DB
make install-locked  # installs the pinned stack + dev tools (what CI tests)
```

Local `pip install -e .` still uses the looser floors so dev installs stay
fast. The lockfile is consumed by the Dockerfile and by CI, which also runs a
non-blocking job against the newest versions as an early warning.

#### Upgrading dependencies

1. `make lock-upgrade`, then `make install-locked`.
2. `make test` and `make audit` must pass. `tests/test_dependency_regression.py`
   checks that checkpoints produced by the previous stack still load and score
   the same, and that a seeded training run stays close to its golden history.
3. Run an example end to end, e.g.
   `PYTHONPATH=src:. python examples/movielens100k_slate/run_training.py`.
4. If the golden history drifted for an expected reason, review it and refresh
   it with `PYTHONPATH=. python scripts/make_regression_fixtures.py --golden-only`. Do not
   regenerate `tests/_fixtures/legacy_artifacts/` to silence a failure.

## Configuration

The framework reads a small set of environment variables. None of them are
required for local training — defaults work out of the box. The Docker /
Vertex AI flow needs GCP-related values; supply them via a local `.env`
file (gitignored) created from the committed `.env.example` template:

```bash
cp .env.example .env
# edit .env and set GCP_PROJECT, GCS_BUCKET
```

The `Makefile` auto-loads `.env` via `-include .env`. You can also override
per-invocation: `make GCP_PROJECT=other-id docker-push`.

| Variable | Default | Sensitivity | Used by |
|---|---|---|---|
| `OUTPUT_PATH` | `output` (or `AIP_MODEL_DIR` if set) | neutral | `utils/files.py` — root for all training artifacts |
| `AIP_MODEL_DIR` | unset | neutral (set by Vertex AI) | `entrypoint.sh`, `utils/files.py` — GCS path for post-training upload |
| `DATA_URI` | unset | depends on URI | `entrypoint.sh` — optional dataset source (`gs://`, `http(s)://`, or local file) |
| `LUIGI_CONFIG_PATH` | auto (package's `luigi.cfg`) | neutral | `cli.py` — internal Luigi config path |
| `GCP_PROJECT` | empty | **sensitive** | `Makefile` — Artifact Registry / Cloud Build / Vertex AI |
| `GCP_REGION` | `us-central1` | neutral | `Makefile` |
| `GCS_BUCKET` | empty | **sensitive** | `Makefile` — Vertex AI artifacts bucket (`gs://...`) |
| `IMAGE_NAME` / `IMAGE_TAG` | `mars-rec` / `latest` | neutral | `Makefile` — Docker image identity |

GCP-touching `make` targets (`docker-push`, `docker-pull`, `gcp-create-repo`,
`gcp-submit`, `vertex-run`) refuse to run when `GCP_PROJECT` / `GCS_BUCKET`
are unset and print the fix.

## Quickstart

### 1. Train one of the bundled recsys examples

```bash
cd mars-rec
PYTHONPATH=src:. python examples/movielens/run_training.py             # MovieLens 1M — NCF (collaborative filtering)
PYTHONPATH=src:. python examples/lastfm/run_training.py                # Last.fm HetRec 2K — sequential (SASRec-lite)
PYTHONPATH=src:. python examples/online_retail/run_training.py         # UCI Online Retail — session-based (GRU4Rec)
PYTHONPATH=src:. python examples/amazon_beauty/run_training.py         # Amazon Beauty 5-core — content-based two-tower
PYTHONPATH=src:. python examples/movielens100k_slate/run_training.py   # MovieLens 100K — slate ranking
```

Each script downloads its dataset on first run (cached under `output/<example>/dataset/`).
See `examples/<name>/data.py` for the dataset URL and preprocessing.

### 2. Train via CLI

```bash
# MovieLens 1M — recommendation with NCF
mars-rec run supervised \
  --project examples.movielens.config.movielens \
  --recommender-module-class examples.movielens.model.MovieLensNCF \
  --recommender-extra-params '{"n_factors": 32, "hidden_size": 128, "n_layers": 3, "dropout": 0.2}' \
  --epochs 10 \
  --batch-size 512 \
  --loss-function bce_logits \
  --negative-proportion 0.5 \
  --dataset-split-method time \
  --device cpu

# Last.fm — sequential next-item with SASRec-lite
mars-rec run supervised \
  --project examples.lastfm.config.lastfm \
  --recommender-module-class examples.lastfm.model.LastFMSASRec \
  --recommender-extra-params '{"n_factors": 64, "n_heads": 2, "ff_dim": 128, "history_len": 10}' \
  --epochs 5 \
  --batch-size 256 \
  --loss-function bce_logits \
  --negative-proportion 0.5 \
  --dataset-split-method time \
  --device cpu

# MovieLens 100K — slate ranking (available_items + metadata)
mars-rec run supervised \
  --project examples.movielens100k_slate.config.movielens100k_slate \
  --recommender-module-class examples.movielens100k_slate.model.ML100KSlateRanker \
  --recommender-extra-params '{"n_factors": 32, "genre_dim": 19, "hidden_size": 64}' \
  --epochs 8 \
  --batch-size 256 \
  --loss-function bce_logits \
  --device cpu

# Force re-execution ignoring cache
mars-rec run supervised --no-cache \
  --project examples.movielens.config.movielens \
  --recommender-module-class examples.movielens.model.MovieLensNCF \
  --epochs 10
```

### 3. Train via Python

```python
import luigi
from mars_rec.simulation.training import SupervisedModelTraining

task = SupervisedModelTraining(
    project="examples.movielens.config.movielens",
    recommender_module_class="examples.movielens.model.MovieLensNCF",
    recommender_extra_params={"n_factors": 32, "hidden_size": 128, "n_layers": 3, "dropout": 0.2},
    epochs=10,
    batch_size=512,
    learning_rate=0.001,
    loss_function="bce_logits",
    negative_proportion=0.5,
    dataset_split_method="time",
    device="cpu",
    early_stopping_patience=3,
    metrics=["loss", "binary_accuracy"],
)

luigi.build([task], local_scheduler=True, log_level="INFO")
```

## Core concepts

### ProjectConfig

`ProjectConfig` is the central object that defines a recommendation project. It specifies:

- **base_dir**: root directory for project data
- **prepare_data_frames_task**: Luigi task class that prepares train/val/test
- **dataset_class**: PyTorch Dataset class to use
- **user_column / item_column**: user and item columns with their types (`IOType`)
- **other_input_columns**: additional features (numeric, arrays, etc.)
- **output_column**: target column (click, rating, etc.)
- **metadata_columns**: item metadata columns

### IOType

Defines how each column is handled in the pipeline:

- `INDEXABLE` — Categorical, converted to integer index (embeddings)
- `NUMBER` — Numeric, used directly as float
- `INDEXABLE_ARRAY` — Array of categorical values, each indexed
- `FLOAT_ARRAY` / `INT_ARRAY` — Numeric arrays

### RecommenderModule

Base class for all models. Extends `nn.Module` and provides:

- `self._n_users` / `self._n_items` — User and item vocabulary sizes
- `self._index_mapping` — Mapping from original values to indices
- `self._project_config` — Project configuration
- `recommendation_score(*args)` — Alias for `forward()`

The `forward()` method receives tensors in the order of the `ProjectConfig`'s `input_columns` (user_ids, item_ids, features...) and must return a tensor of shape `[batch_size]` with prediction scores.

### BasePrepareDataFrames

Luigi task that orchestrates data preparation:

1. Reads the interactions DataFrame
2. Applies filters (`eq_filters`, `neq_filters`, `isin_filters`)
3. Generates the `available_arms` column (if not present)
4. Splits into train/val/test (methods: `time`, `holdout`, `column`, `k_fold`)
5. Applies balancing (oversample/undersample) if configured
6. Saves output CSVs

### SupervisedModelTraining

Luigi task that executes the full training pipeline:

1. Loads prepared DataFrames (train/val/test)
2. Creates index mapping (categoricals → integers)
3. Instantiates the model (`RecommenderModule`) within a `RecommenderLightningModule`
4. Trains with PyTorch Lightning (callbacks: EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard)
5. Evaluates on the validation set
6. Generates predictions on the test set with score-based ranking
7. Saves artifacts: `weights.ckpt`, `history.csv`, `params.json`, `test_set_predictions.csv`, `plot_scores.png`, `plot_history.jpg`

## Training parameters

| Parameter | Default | Description |
|---|---|---|
| `project` | (required) | Path to ProjectConfig (e.g., `examples.movielens.config.movielens`) |
| `recommender_module_class` | (required) | Path to the model class |
| `recommender_extra_params` | `{}` | Extra params passed to the model constructor |
| `epochs` | `100` | Maximum number of epochs |
| `batch_size` | `500` | Batch size |
| `learning_rate` | `0.001` | Learning rate |
| `optimizer` | `adam` | Optimizer (`adam`, `sgd`, `rmsprop`, `radam`, `adadelta`, `adagrad`, `adamax`) |
| `optimizer_params` | `{}` | Extra kwargs forwarded to the optimizer constructor |
| `loss_function` | `bce` | Loss function (`bce`, `bce_logits`, `mse`, `nll`, `ce`, `mlm`, `focal_loss`, `implicit_feedback_bce`, `dummy`) |
| `loss_function_params` | `{}` | Extra kwargs forwarded to the loss constructor |
| `device` | auto | `cpu`, `cuda` or `mps` (auto-detected) |
| `early_stopping_patience` | `5` | Epochs without improvement before stopping |
| `early_stopping_min_delta` | `0.001` | Minimum delta considered as improvement |
| `monitor_metric` | `val_loss` | Metric monitored for early stopping / checkpointing |
| `monitor_mode` | `min` | `min` or `max` |
| `test_size` | `0.2` | Test set proportion |
| `val_size` | `0.2` | Validation set proportion |
| `dataset_split_method` | `time` | Split method (`time`, `holdout`, `column`, `k_fold`) |
| `test_split_type` | `random` | Test split strategy (`random`, `time`) |
| `negative_proportion` | `0.0` | Negative sampling proportion (train/val) |
| `sampling_strategy` | `none` | Balancing strategy (`none`, `oversample`, `undersample`) |
| `balance_fields` | `[]` | Columns used for balancing (overrides project default) |
| `sampling_proportions` | `{}` | Per-class proportions for over/undersampling |
| `eq_filters` / `neq_filters` / `isin_filters` | `{}` | Row filters applied during data preparation |
| `data_frames_preparation_extra_params` | `{}` | Extra kwargs forwarded to the prepare-data-frames task |
| `metrics` | `["loss"]` | Metrics tracked during training (e.g. `binary_accuracy`, `f1_score`, `accuracy`, `ndcg@10`, `mrr@10`) |
| `ranking_metrics` | `False` | Compute NDCG / HitRate / MRR / MAP (@5, 10, 20) on the test set |
| `sample_size_eval` | `None` | Cap the number of test rows used for evaluation |
| `gradient_norm_clipping` | `0.0` | Gradient clipping (0 = disabled) |
| `gradient_norm_clipping_type` | `2` | Norm type for gradient clipping |
| `generator_workers` | `0` | DataLoader `num_workers` |
| `pin_memory` | `False` | DataLoader `pin_memory` |
| `seed` | `42` | Random seed |
| `observation` | `""` | Free-form note (becomes part of the task hash → useful to force a re-run) |
| `load_index_mapping_path` | `None` | Reuse an existing `index_mapping.pkl` from another run |

## CLI commands

### `mars-rec run supervised`

Runs the full training pipeline (data preparation + training + evaluation).

```bash
mars-rec run supervised \
  --project examples.movielens.config.movielens \
  --recommender-module-class examples.movielens.model.MovieLensNCF \
  --epochs 10 --device cpu
```

Use `--no-cache` to force re-execution of the entire pipeline, ignoring previous results:

```bash
mars-rec run supervised --no-cache --project ...
```

### `mars-rec list`

Lists all executed training runs, with date, project, model, and task ID.

```bash
# List all training runs
mars-rec list

# Filter by project
mars-rec list --project movielens

# Limit results
mars-rec list --limit 10
```

Example output:

```
#    Date               Project     Model               Epochs   Task ID
----------------------------------------------------------------------------------
1    2026-05-16 12:55   lastfm      LastFMSASRec        5        SupervisedModelTraining____256____f37b2b1e04
2    2026-05-16 13:33   ml100k_slt  ML100KSlateRanker   8        SupervisedModelTraining____256____1ab7b0e50e
3    2026-03-28 16:24   synthetic   SimpleMLP           10       SupervisedModelTraining____128____28023c8618
```

### `mars-rec results`

Compares metrics and hyperparameters across multiple runs. Useful for experiment tracking.

```bash
# View all results
mars-rec results

# Filter by project and sort by best val_loss
mars-rec results --project lastfm --sort-by val_loss

# Select specific metrics
mars-rec results --metric val_loss --metric val_binary_accuracy

# Add hyperparameters as columns (also searches recommender_extra_params)
mars-rec results --params dropout --params n_factors --params hidden_size

# Combine filters
mars-rec results --project lastfm --sort-by val_loss --params dropout --limit 10
```

Example output:

```
               date  project  model            epochs  lr     batch_size  loss_fn      val_binary_accuracy  val_loss
1  2026-05-16 13:33  movielens100k_slate  ML100KSlateRanker  8       0.001  256         bce_logits  0.799278             0.406891
2  2026-05-16 12:55  lastfm               LastFMSASRec       5       0.001  256         bce_logits  0.582535             0.675994
```

**`results` options:**

| Option | Description |
|---|---|
| `--project TEXT` | Filter by project name (substring match) |
| `--metric TEXT` | Metrics to display (repeatable). Shows all if omitted |
| `--sort-by TEXT` | Sort results by this metric (ascending) |
| `--params TEXT` | Extra parameters to display as columns (repeatable). Searches in `params` and `recommender_extra_params` |
| `--limit INT` | Maximum number of runs to display (default: 20) |

### `mars-rec predict`

Runs batch inference on a trained model and writes a JSON file with the top-k items per user.

```bash
# By task ID (auto-resolves to output/models/SupervisedModelTraining/results/<task_id>)
mars-rec predict \
  --task-id SupervisedModelTraining____64____a0cdf4af5e \
  --data data/test.parquet \
  --output result.json \
  --top-k 10 --batch-size 128 --device cpu

# By full directory path
mars-rec predict --model-dir output/models/.../<task_id> --data data/test.parquet
```

Output `result.json` schema:

```json
{
  "top10s": [[item_id, item_id, ...], ...],
  "user":   [user_id, user_id, ...],
  "time":   12.34
}
```

**`predict` options:**

| Option | Description |
|---|---|
| `--model-dir PATH` | Full path to the model task directory (mutually exclusive with `--task-id`) |
| `--task-id TEXT` | Task ID; resolved inside the standard results directory |
| `--data PATH` | **Required.** Input data file (`.parquet` or `.csv`) |
| `--output PATH` | Output JSON path (default: `result.json`) |
| `--batch-size INT` | Batch size for inference (default: 128) |
| `--top-k INT` | Number of top items per user (default: 10) |
| `--device TEXT` | Device: `cpu`, `cuda`, `cuda:0`, ... (default: `cpu`) |

### `mars-rec evaluate`

Re-runs evaluation on an already-trained model. Two flows:

- **Flow 1 — original splits**: re-evaluate val/test CSVs from the training run.
- **Flow 2 — external dataset**: evaluate any CSV/parquet that contains the target column (preprocessed with the same pipeline).

```bash
# Flow 1 — re-evaluate original splits, enable ranking metrics + override metrics
mars-rec evaluate \
  --task-id SupervisedModelTraining____64____a0cdf4af5e \
  --ranking-metrics --metric loss --metric f1_score --metric binary_accuracy \
  --split test

# Flow 2 — evaluate a new dataset
mars-rec evaluate \
  --task-id SupervisedModelTraining____64____a0cdf4af5e \
  --data new_test.parquet \
  --ranking-metrics
```

Writes ranking metrics (NDCG, HitRate, MRR, MAP @{5,10,20}) and ROC-AUC into `metadata.json` inside the task dir (read by `list` / `results`).

**`evaluate` options:**

| Option | Description |
|---|---|
| `--model-dir PATH` | Full path to the model task directory |
| `--task-id TEXT` | Task ID resolved inside the standard results directory |
| `--data PATH` | External CSV/parquet for Flow 2; if omitted uses original splits |
| `--metric TEXT` | Override validation metrics (repeatable) |
| `--ranking-metrics / --no-ranking-metrics` | Enable/disable NDCG, HitRate, MRR, MAP |
| `--sample-size-eval INT` | Override test-set sample size |
| `--batch-size INT` | Override batch size |
| `--device TEXT` | Override device (`cpu`, `cuda`, `mps`) |
| `--split [val\|test\|all]` | Which split to evaluate (Flow 1 only; default `all`) |
| `--output PATH` | Directory to save eval artifacts (default: model task dir) |

### `mars-rec run data`

Runs an arbitrary Luigi data-preparation task (anything inheriting from `BasePrepareDataFrames`). Useful to materialize the train/val/test CSVs without launching a training:

```bash
mars-rec run data examples.movielens.data PrepareMovieLensDataFrame
```

## Output

After training, artifacts are saved to `output/models/SupervisedModelTraining/results/<task_id>/`:

```
├── params.json              # Hyperparameters used
├── weights.ckpt             # PyTorch Lightning checkpoint (model weights)
├── history.csv              # Per-epoch metrics (loss, accuracy, etc.)
├── metadata.json            # Post-training ranking + ROC-AUC metrics (when computed)
├── plot_history.jpg         # Training curves plot
├── plot_scores.png          # Prediction score distribution
├── summary.txt              # Model architecture (layers, params)
├── sample_train.csv         # Training data sample
├── index_mapping.pkl        # Index mapping (for inference)
├── test_set_predictions.csv # Test set predictions with ranking
└── lightning_logs/          # PyTorch Lightning logs
    └── metrics.csv          # Raw per-step metrics
```

`metadata.json` is consumed by `mars-rec list` and `mars-rec results` so that ranking and ROC-AUC metrics show up alongside training metrics.

## Creating your own project

### 1. Define data preparation

```python
# my_project/data.py
import luigi
import pandas as pd
from mars_rec.data.task import BasePrepareDataFrames

class PrepareMyDataFrame(BasePrepareDataFrames):
    def requires(self):
        return MyDownloadTask()  # or luigi.ExternalTask

    @property
    def dataset_dir(self) -> str:
        return "output/my_project/dataset"

    @property
    def read_data_frame_path(self) -> str:
        return self.input().path

    @property
    def timestamp_property(self) -> str:
        return "timestamp"  # timestamp column for temporal split

    def transform_data_frame(self, df, data_key):
        # Custom transformations (optional)
        df["feature"] = df["feature"].astype(float)
        return df
```

### 2. Configure the project

```python
# my_project/config.py
from mars_rec.meta_config import ProjectConfig, Column, IOType
from mars_rec.data.dataset import InteractionsDataset
from my_project.data import PrepareMyDataFrame

my_project = ProjectConfig(
    base_dir="output/my_project",
    prepare_data_frames_task=PrepareMyDataFrame,
    dataset_class=InteractionsDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    other_input_columns=[
        Column("price", IOType.NUMBER),
    ],
    output_column=Column("purchased", IOType.NUMBER),
)
```

### 3. Implement the model

```python
# my_project/model.py
import torch
import torch.nn as nn
from mars_rec.model.abstract import RecommenderModule

class MyModel(RecommenderModule):
    def __init__(self, project_config, index_mapping, n_factors=32):
        super().__init__(project_config, index_mapping)
        self.user_emb = nn.Embedding(self._n_users, n_factors)
        self.item_emb = nn.Embedding(self._n_items, n_factors)
        self.fc = nn.Linear(n_factors * 2 + 1, 1)  # +1 for "price"

    def forward(self, user_ids, item_ids, price):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i, price.unsqueeze(1)], dim=1)
        return torch.sigmoid(self.fc(x)).squeeze(-1)  # shape: [batch_size]
```

**Important**: `forward()` must return a tensor of shape `[batch_size]` (no extra dimension). Use `.squeeze(-1)` if the last layer returns `[batch_size, 1]`.

### 4. Run training

```bash
mars-rec run supervised \
  --project my_project.config.my_project \
  --recommender-module-class my_project.model.MyModel \
  --recommender-extra-params '{"n_factors": 32}' \
  --epochs 20
```

## Included examples

Each example is a self-contained recommendation project and lives in `examples/<name>/`
with four files: `data.py` (Luigi download + preprocess), `config.py` (`ProjectConfig`),
`model.py` (`RecommenderModule` subclass), and `run_training.py`.

| Example | Dataset | Framework feature it exercises |
|---|---|---|
| `movielens` | MovieLens 1M | Implicit-feedback CF with side features (gender/age/occupation + genre multi-hot) |
| `lastfm` | HetRec Last.fm 2K | Sequence column via `IOType.INDEXABLE_ARRAY` + `same_index_as` (SASRec-lite) |
| `online_retail` | UCI Online Retail | `user_column` bound to `session_id` — session-based GRU4Rec without stable user identity |
| `amazon_beauty` | Amazon Beauty 5-core | `metadata_columns` with `FLOAT_ARRAY` text embedding (TF-IDF + SVD), `CONTENT_BASED` two-tower |
| `movielens100k_slate` | MovieLens 100K + synthetic slates | `available_items_column_name` + `InteractionsWithNegativeItemGenerationByAvailableItemsDataset` + `metadata_columns` |

### movielens (recommendation — requires download)

MovieLens 1M dataset from GroupLens (~1M ratings from 6,040 users on 3,706 movies). Implicit feedback: rating >= 4 is positive. User features (gender, age, occupation) and movie features (18 multi-hot genres). Automatically downloaded from https://files.grouplens.org/datasets/movielens/ml-1m.zip.

Model `MovieLensNCF` (Neural Collaborative Filtering): combines a GMF path (element-wise product of embeddings) with an MLP path that incorporates user and genre features. Uses `BCEWithLogitsLoss` with negative sampling (`negative_proportion=0.5`) and temporal split.

```bash
# Via standalone script
PYTHONPATH=src:. python examples/movielens/run_training.py

# Via CLI
mars-rec run supervised \
  --project examples.movielens.config.movielens \
  --recommender-module-class examples.movielens.model.MovieLensNCF \
  --recommender-extra-params '{"n_factors": 32, "hidden_size": 128, "n_layers": 3, "dropout": 0.2}' \
  --epochs 10 --batch-size 512 --loss-function bce_logits \
  --negative-proportion 0.5 --dataset-split-method time --device cpu
```

### lastfm (sequential next-item — requires download)

HetRec 2011 Last.fm 2K dataset (~76K tagging events after filtering, 1.2K users, 6.8K artists), used as an ordered click-stream per user with a fixed-length history window.

Model `LastFMSASRec`: shared item embedding + positional emb → 1-layer causal Transformer over `history_artist_ids` → dot product against target item.

```bash
PYTHONPATH=src:. python examples/lastfm/run_training.py
```

### online_retail (session-based — requires download)

UCI Online Retail (~510K purchase line-items, 18K invoices, 3.5K stock codes). Each invoice (`InvoiceNo`) is treated as a session — there is no stable user_id, demonstrating that `user_column` can bind to any indexable entity.

Model `OnlineRetailGRU4Rec`: shared item embedding → 1-layer GRU over per-session history → dot product against target item.

```bash
PYTHONPATH=src:. python examples/online_retail/run_training.py
```

### amazon_beauty (content-based — requires download)

Amazon Beauty 5-core reviews (~198K reviews, 22K users, 12K items). Per-item text (review summaries + bodies) is embedded with TF-IDF + TruncatedSVD into a 64-d vector and stored in a separate metadata CSV looked up by item index at sample time (`metadata_columns`).

Model `AmazonBeautyTwoTower`: user-id embedding → MLP user tower; precomputed text embedding → MLP item tower; dot-product score. The item tower has no per-item parameter, so the model generalises to cold items.

```bash
PYTHONPATH=src:. python examples/amazon_beauty/run_training.py
```

### movielens100k_slate (slate ranking — requires download)

MovieLens 100K with synthetic per-row candidate slates (1 positive + 49 random items). Negatives are sampled from the slate via `InteractionsWithNegativeItemGenerationByAvailableItemsDataset` rather than uniformly from the full catalogue — matching the "leave-one-out + 99 random negatives" benchmark format. 19-dim binary genre vector per movie via `metadata_columns`.

Model `ML100KSlateRanker`: two-tower with user-id embedding and (item-id ⊕ genre) embedding, dot product.

```bash
PYTHONPATH=src:. python examples/movielens100k_slate/run_training.py
```

## Docker / Vertex AI

Common Docker and GCP workflows are exposed via the `Makefile`:

```bash
# Build the image locally (linux/amd64; default base = pytorch/pytorch:2.14.1-cuda12.6)
make docker-build

# Run a training inside the container
make docker-run ARGS="--project examples.movielens.config.movielens \
  --recommender-module-class examples.movielens.model.MovieLensNCF --epochs 5 --device cpu"

# Push to Google Artifact Registry (configures via GCP_PROJECT / GCP_REGION env vars)
make gcp-setup
make docker-push

# Submit a Vertex AI Custom Training job
make vertex-run ARGS="--project examples.movielens.config.movielens \
  --recommender-module-class examples.movielens.model.MovieLensNCF \
  --epochs 10 --batch-size 256 --device cuda:0 --no-cache"
```

Run `make help` to list all targets.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the branch, lint and pull request workflow.

## License

MIT
