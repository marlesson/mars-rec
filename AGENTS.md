# AGENTS.md — Guide for AI Agents

This document guides AI agents (Claude Code, Copilot, Cursor, etc.) on how to navigate, modify, and extend the MARS-Rec project.

## Language standard

**All documentation, code comments, commit messages, and PR descriptions in this project MUST be written in English.** This is an international project and English is the standard language across all artifacts.

## Work in a git worktree (mandatory)

**Every task that modifies files MUST be done in a dedicated git worktree, never directly in the main checkout.** The main checkout may hold someone else's in-progress work (or another agent's); a worktree isolates each task on its own branch.

1. Create the worktree under `.claude/worktrees/` (gitignored), branching from an up-to-date `main` and using the branch prefixes from `CONTRIBUTING.md`:
   ```bash
   git fetch origin main
   git worktree add .claude/worktrees/<short-name> -b <type>/<short-name> origin/main
   cd .claude/worktrees/<short-name>
   ```
   Claude Code agents can use the built-in `EnterWorktree` tool instead, which does the same thing.
2. Run every command (edits, `make lint`, `make test`, git) from inside the worktree. Do not `cd` back to the main checkout to work.
3. **Set `PYTHONPATH=src:.`** for anything that imports `mars_rec`. The shared `.venv` has an *editable install pointing at the main checkout's `src/`*, so without it Python silently loads the main checkout's code, not the worktree's:
   ```bash
   PYTHONPATH=src:. make test
   PYTHONPATH=src:. python -c "import mars_rec; print(mars_rec.__file__)"  # must print a path inside the worktree
   ```
4. Training outputs default to `output/` relative to the current directory, so runs started in a worktree stay in that worktree. Set `OUTPUT_PATH` explicitly to reuse cached datasets/runs from the main checkout.
5. Do not use bare `git stash` / `git stash pop`: the stash stack is shared across all worktrees. Use a temporary WIP commit instead.
6. When the branch is merged (or abandoned), remove the worktree: `git worktree remove .claude/worktrees/<short-name>` and delete the branch.

Read-only tasks (answering questions, reviewing code, exploring) do not need a worktree.

## Project overview

MARS-Rec is a framework for training deep learning models for recommendation systems. It uses Luigi for pipeline orchestration, PyTorch for models, and PyTorch Lightning for the training loop. It contains no Reinforcement Learning, bandits, or exploration policy code.

**Scope rule**: this repository holds only the framework (`src/mars_rec/`) and the reference projects under `examples/`. Application / competition / research code (custom models, sweeps, submission pipelines) does not belong here — keep it in a separate project that depends on `mars-rec` as a library. Framework code must stay domain-agnostic: no hard-coded column names, dataset names or competition-specific logic in `src/`.

## Directory structure

```
src/mars_rec/          # Framework source code (the only installable package)
  meta_config.py            # ProjectConfig, Column, IOType — central definitions
  cli.py                    # CLI (Click): "run supervised", "run data", "list", "results", "predict", "evaluate", "visualize"
  filtering.py              # CandidateProvider / CandidateFilter — inference-time candidate pools
  inference.py              # Standalone inference + evaluation (run_inference, run_evaluate)
  visualization.py          # Architecture visualization — single module, SVG-only.
                            # Exposes generate_architecture_diagram(), trace_model_shapes(),
                            # extract_model_ir(), render_svg_flow(), load_model_from_task(),
                            # print_text_summary(). No external system deps.
  luigi.cfg                 # Default Luigi configuration
  data/
    dataset.py              # PyTorch Datasets (InteractionsDataset and negative sampling variants)
    task.py                 # BasePrepareDataFrames — Luigi task for train/val/test split
    utils.py                # DownloadDataset helper
  model/
    abstract.py             # RecommenderModule — base class for ALL models
    base_model.py           # LogisticRegression — baseline model
  simulation/
    training.py             # SupervisedModelTraining + run_evaluate — MAIN FILE (~2.4k lines)
  torch/
    callbacks.py            # Lightning callbacks
    data.py                 # Custom DataLoader and BatchSampler
    init.py                 # Weight initializers
    lightning_module.py     # RecommenderLightningModule (Lightning wrapper)
    loss.py                 # Custom loss functions
    metrics.py              # Evaluation metrics (METRIC_REGISTRY)
    optimizer.py            # RAdam optimizer
    summary.py              # Model summary print
  utils/
    files.py                # Output artifact path functions (OUTPUT_PATH env var supported)
    index_mapping.py        # Categorical → integer mapping for embeddings
    logging.py              # Logger setup
    reflection.py           # load_attr — dynamic module import by string
    utils.py                # General utilities (parallel literal_eval, reduce memory)
    download.py             # download_file — stdlib HTTP download (certifi CA bundle, atomic rename)
    plot.py                 # Training history plots

examples/                   # Reference recsys projects (public datasets, auto-download)
  movielens/                # MovieLens 1M — NCF (collaborative filtering + side features)
  lastfm/                   # Last.fm HetRec 2K — sequential SASRec-lite
  online_retail/            # UCI Online Retail — session-based GRU4Rec (session_id as user)
  amazon_beauty/            # Amazon Beauty 5-core — content-based two-tower (text emb)
  movielens100k_slate/      # MovieLens 100K — slate ranking (available_items + metadata)

docs/                       # Documentation site (MkDocs Material) — published to GitHub Pages
  _ext/griffe_sphinx_roles.py  # Docs-only griffe extension: Sphinx roles → Markdown links
  reference/api/*.md        # mkdocstrings stubs (`::: mars_rec.<module>`) — API pages are generated
mkdocs.yml                  # Site config + nav (strict build)

scripts/
  make_vertex_config.py     # Builds the Vertex AI job config used by `make vertex-run`
  make_regression_fixtures.py  # Regenerates tests/_fixtures/{legacy_artifacts,golden_history.json}

tests/                      # Pytest suite (test_dataset, test_loss, test_metrics, test_meta_config, ...)
CONTRIBUTING.md             # Branch / lint / PR workflow
.github/                    # CI + Docs + Release (PyPI, on v* tags) workflows, Dependabot, PR template
Makefile                    # Development / Docker / Vertex AI targets (`make lint` = ruff check + format --check)
Dockerfile                  # CPU/GPU image (default base: pytorch/pytorch:2.14.1-cuda12.6)
entrypoint.sh               # Container entrypoint (dataset download, Vertex AI artifact upload)
pyproject.toml              # Package metadata + dependencies; requirements.lock pins them
```

## Execution flow

The pipeline follows this sequence orchestrated by Luigi:

```
1. DataTask (e.g., GenerateSyntheticData)
   └── Generates or downloads raw data
       │
2. BasePrepareDataFrames (e.g., PrepareSyntheticDataFrame)
   └── Reads CSV → filters → splits train/val/test → saves 3 CSVs
       │
3. SupervisedModelTraining
   ├── Loads train/val/test DataFrames
   ├── Creates index_mapping (categorical → integer)
   ├── Instantiates the model (RecommenderModule) within RecommenderLightningModule
   ├── Trains with PyTorch Lightning (callbacks: EarlyStopping, Checkpoint, CSVLogger, TensorBoard if the `tensorboard` extra is installed)
   ├── Evaluates on the validation set
   ├── Generates predictions on the test set
   ├── Optional: ranking metrics (NDCG/HitRate/MRR/MAP @{5,10,20}) + ROC-AUC → metadata.json
   └── Saves artifacts: params.json + metadata.json at the root, model/ (weights.ckpt, index_mapping.pkl, …),
       training/ (history.csv, plots, samples) and eval/test/ (metrics.json, predictions.csv, plots/)
```

After training, two standalone entry points reuse the same artifacts:

- **`run_inference`** (`inference.py`) — top-k predictions from a `params.json` + `weights.ckpt` + `index_mapping.pkl`, exposed as `mars-rec predict`.
- **`run_evaluate`** (`simulation/training.py`) — re-runs evaluation on the original test split or on an external CSV/parquet, exposed as `mars-rec evaluate`.

## Key concepts the agent MUST understand

### 1. ProjectConfig is the central contract

All configuration for a recommendation project lives in `ProjectConfig`. It defines:
- Which columns exist and their types (`IOType`)
- Which task prepares the data
- Which dataset class to use

**Location**: `src/mars_rec/meta_config.py`

When creating a new project, ALWAYS start with the `ProjectConfig`.

### 2. Dynamic class resolution by string

The framework uses strings to reference classes (via `load_attr` in `utils/reflection.py`):

```python
# Example: "examples.movielens.model.MovieLensNCF"
# Is resolved to the MovieLensNCF class at runtime
```

This means `project` and `recommender_module_class` are **importable Python paths** (dotted notation). The `PYTHONPATH` must include the project root directory.

### 3. forward() argument order

The model's `forward()` method receives tensors in the **same order** as the columns defined in `ProjectConfig.input_columns`:

```python
# If ProjectConfig has:
#   user_column = Column("user_id", IOType.INDEXABLE)
#   item_column = Column("item_id", IOType.INDEXABLE)
#   other_input_columns = [Column("price", IOType.NUMBER)]
# Then forward receives:
def forward(self, user_ids, item_ids, price):
```

### 4. forward() output shape

The expected `forward()` output depends on the loss function:

| Loss | Output shape | Notes |
|---|---|---|
| `bce`, `implicit_feedback_bce` | `[batch_size]` | Probabilities (apply `sigmoid` in the model) |
| `bce_logits`, `bce_logits_smooth`, `focal_loss` | `[batch_size]` | Raw logits (in `_LOGIT_LOSSES`; metrics get `sigmoid(score)`) |
| `bpr`, `ranknet`, `lambdarank` | `[batch_size]` | Raw logits; pairwise/listwise over the batch |
| `mse`, `dummy` | `[batch_size]` | One score per row |
| `ce`, `nll` | `[batch_size, n_items]` | Scores over the item vocabulary (next-item / sequence models) |
| `infonce`, `cosine_similarity`, `hardneg_infonce` | embeddings | `EmbeddingRecommenderLightningModule` |

The user-facing version of this table lives in `docs/concepts/models.md` — keep both in sync.

For `[batch_size, 1]` outputs, add `.squeeze(-1)`:

```python
def forward(self, user_ids, item_ids):
    ...
    return torch.sigmoid(score).squeeze(-1)  # [batch, 1] → [batch]
```

For CE-based next-item prediction, return logits without squeezing:

```python
def forward(self, *features):
    ...
    return self.head(hidden)  # [batch, n_items]
```

Mismatched shapes are the most common source of `RuntimeError: target size != input size`.

### 5. Index mapping

Columns of type `IOType.INDEXABLE` are converted from categorical values to integers before being passed to the model. The mapping is automatically created by `SupervisedModelTraining` and saved to `index_mapping.pkl`.

- Index `0` is reserved for unknown values (never seen during training)
- Index `1` is reserved for `None` / `NaN`
- Index `2` is reserved for `-1` / `"-1"` (the pad token for sequence columns)
- Actual values start at `3`

Inference never returns a reserved slot as a recommendation: `build_reverse_item_mapping` drops them and `predict_top_k` masks their logits.

### 6. Luigi caching and --no-cache

Luigi marks tasks as DONE based on the existence of output files. For `SupervisedModelTraining` the output is the `<task_id>/_SUCCESS` marker, written as the very last step of `run()`, so a run killed midway is re-executed instead of being treated as done. Use `task.task_dir` (not `task.output().path`) for the run directory. Legacy flat-layout runs without the marker count as done when they have a checkpoint and `metadata.json` (`utils/files.py::is_complete_run`). To force re-execution of a finished run:

- Use `--no-cache` in the CLI: `mars-rec run supervised --no-cache ...`
- Manually delete the output directory, OR
- Change a task parameter (generates a new task_id hash)

`--no-cache` works by calling `task.invalidate_cache()` which deletes the data preparation and training outputs before starting Luigi, forcing re-execution of the entire pipeline.

### 7. CLI commands

The CLI exposes seven commands rooted at `mars-rec`:

| Command | Purpose |
|---|---|
| `run supervised [--no-cache]` | Full training pipeline (data prep + train + eval). `--no-cache` calls `task.invalidate_cache()` to force re-execution. |
| `run data <module> <Task>` | Run an arbitrary Luigi task (e.g. just the data preparation). |
| `list [--project] [--limit]` | List executed runs (date, project, model, epochs, task_id). |
| `results [--project] [--metric] [--sort-by] [--params] [--limit]` | Compare metrics across runs; reads `params.json` + `history.csv` + `metadata.json`. |
| `predict --task-id … --data … [--top-k] [--device]` | Standalone batch inference → `result.json` with top-k items per user. Implemented in `inference.run_inference`. |
| `evaluate --task-id … [--data] [--name] [--metric] [--ranking-metrics] [--split]` | Re-run evaluation on original splits (Flow 1) or on an external CSV/parquet (Flow 2). Implemented in `simulation.training.run_evaluate`. Each evaluation writes to `eval/<name>/` (`test` for Flow 1, the `--data` file stem for Flow 2); only `eval/test` refreshes `metadata.json`, which is what `list/results` read. |
| `visualize --task-id … [--mode {svg,summary}] [--trace/--no-trace] [--output]` | Generate the architecture diagram. `svg` (default) writes a hand-rolled SVG flow diagram with role colors + feature pills + tensor shapes. `summary` prints a text table. `--trace` (default on) runs a forward pass with synthetic inputs to capture real tensor shapes via hooks. The legacy `block`/`torchviz` modes and the `--format` flag were removed in the visualization refactor — every training run now emits `architecture.svg` next to the checkpoint automatically. |

All `list/results` lookups read `output/models/SupervisedModelTraining/results/<task_id>/`. The layout inside a task directory (`model/`, `training/`, `eval/<name>/`) is defined only in `utils/files.py` — build artifact paths with its helpers, never by joining file names; old flat-layout runs are still readable through the same helpers.

## Patterns for common modifications

### Adding a new model

1. Create a file at `examples/my_project/model.py`
2. Inherit from `RecommenderModule`
3. Implement `__init__(self, project_config, index_mapping, **kwargs)` and `forward()`
4. `forward()` must return shape `[batch_size]` (`[batch_size, n_items]` for `ce` / `nll`)
5. Pass extra params via `recommender_extra_params`

### Adding a new dataset

1. Create `examples/my_project/data.py` with a class inheriting `BasePrepareDataFrames`
2. Implement: `requires()`, `dataset_dir`, `read_data_frame_path`
3. Optional: `timestamp_property`, `metadata_data_frame_path`, `transform_data_frame()`
4. Create `examples/my_project/config.py` with a `ProjectConfig`

### Adding a new loss function

1. Add the class to `src/mars_rec/torch/loss.py`
2. Register it in the `TORCH_LOSS_FUNCTIONS` dictionary in `simulation/training.py`
3. The loss must accept `(input, target)` and return a scalar

### Adding a new metric

1. Add the function to `src/mars_rec/torch/metrics.py`
2. Register it in `METRIC_REGISTRY` (same file) — the registry key is the name used in the `metrics` list

### Adding a new optimizer

1. Add it to `src/mars_rec/torch/optimizer.py`
2. Register it in the `TORCH_OPTIMIZERS` dictionary in `simulation/training.py`

### Adding a sequence model (next-item prediction)

1. Inherit from `RecommenderModule`; accept `*features` in `forward()` (positional unpacking of `ProjectConfig.input_columns`).
2. Declare the history as an `IOType.INDEXABLE_ARRAY` column with `same_index_as` the item column (see `examples/lastfm/`).
3. Return logits over the item vocabulary: `[batch_size, n_items]` and train with `--loss-function ce`.

### Modifying the inference / evaluate pipeline

- `inference.py::run_inference` is the source of truth for batch top-k prediction. It loads `params.json` + `weights.ckpt` + `index_mapping.pkl`, preprocesses the input the same way training does, and writes `{top10s, user, time}` to JSON.
- `simulation/training.py::run_evaluate` re-builds a `SupervisedModelTraining` instance from `params.json`, points it to the existing task dir (`task._source_task_dir`), and runs the same `evaluate()` method as the original training. Two flows: original splits (`data_path=None`) or external dataset.
- `run_evaluate` writes `eval/<name>/{metrics.json,predictions.csv,plots/}` and refreshes `metadata.json` (what `list` / `results` read for ranking + ROC-AUC metrics) only for the canonical `eval/test` evaluation, so external-data evaluations never overwrite the training-time numbers.

## Documentation

The public docs live in `docs/` and are built with MkDocs Material + mkdocstrings (`make docs`, strict;
`make docs-serve` for live preview; install with `pip install -e ".[docs]"`). When you change
user-facing behavior (CLI options, training parameters, `ProjectConfig` fields, registries), update
the matching page under `docs/` in the same PR. The API reference is generated from docstrings, so keep
docstring parameters in sync with signatures — the strict build fails on mismatches. The example
tutorials include source files via `--8<--` snippets, so they never need manual syncing.
`CONTRIBUTING.md` is also included into the site (`docs/contributing.md`), so links in it must be absolute
GitHub URLs — a relative link such as `(AGENTS.md)` fails the strict build. Pushes to `main` publish to GitHub
Pages (https://marlesson.github.io/mars-rec/) while the `DOCS_DEPLOY` repository variable is `true`.

## Critical files (modify with care)

| File | Risk | Reason |
|---|---|---|
| `simulation/training.py` | **High** | Full training pipeline + `run_evaluate`. ~2.4k lines. Changes can break the entire flow. |
| `meta_config.py` | **High** | Central contract. Changes affect all projects and the dataset/training. |
| `inference.py` | **High** | Standalone inference + evaluation. Must stay aligned with training's preprocessing/index_mapping. |
| `data/dataset.py` | **Medium** | Dataset classes used by every project. Changes to `__getitem__` affect training. |
| `data/task.py` | **Medium** | Data split and preparation. Changes alter how data is divided. |
| `utils/index_mapping.py` | **Medium** | Index mapping. Errors here cause embedding lookup failures. |
| `cli.py` | **Low** | Thin Click wrappers around `run_inference` / `run_evaluate` / Luigi. |

## Global dictionaries in simulation/training.py

These dictionaries map configuration strings to Python classes:

```python
TORCH_OPTIMIZERS = {
    "adam": Adam, "adamw": AdamW, "sgd": SGD, "rmsprop": RMSprop,
    "radam": RAdam, "adadelta": Adadelta, "adagrad": Adagrad, "adamax": Adamax
}

TORCH_LR_SCHEDULERS = {
    "reduce_on_plateau": ReduceLROnPlateau, "cosine_annealing": CosineAnnealingLR,
    "cosine_warm_restarts": CosineAnnealingWarmRestarts, "step": StepLR
}

TORCH_LOSS_FUNCTIONS = {
    "bce": BCELoss, "mse": MSELoss, "nll": NLLLoss, "ce": CrossEntropyLoss,
    "bce_logits": BCEWithLogitsLoss, "bce_logits_smooth": BCEWithLogitsLossSmooth,
    "bpr": BPRLoss, "ranknet": RankNetLoss, "lambdarank": LambdaRankLoss,
    "mlm": MultiLabelMarginLoss, "focal_loss": FocalLoss,
    "implicit_feedback_bce": ImplicitFeedbackBCELoss, "dummy": DummyLoss,
    # Embedding losses (EmbeddingRecommenderLightningModule):
    "infonce": InfoNCELoss, "cosine_similarity": CosineSimilarityLoss, "hardneg_infonce": HardNegInfoNCELoss
}
# Losses that consume logits must also be listed in torch/lightning_module.py::_LOGIT_LOSSES.

TORCH_ACTIVATION_FUNCTIONS = {
    "relu": F.relu, "selu": F.selu, "tanh": F.tanh, "sigmoid": F.sigmoid, "linear": F.linear
}

TORCH_WEIGHT_INIT = {
    "lecun_normal": lecun_normal_init, "he": he_init, "xavier_normal": xavier_normal
}
```

## Environment variables

Read by the code:

| Variable | Default | Usage |
|---|---|---|
| `OUTPUT_PATH` | `"output"` (falls back to `AIP_MODEL_DIR`) | Root directory for all artifacts (`utils/files.py`) |
| `AIP_MODEL_DIR` | unset (Vertex AI sets it automatically) | GCS path for post-training upload; also a fallback for `OUTPUT_PATH` |
| `DATA_URI` | unset | Optional dataset source consumed by `entrypoint.sh` (`gs://`, `http(s)://`, or local path) |
| `DATASET_GCS_PREFIX` | unset | Optional GCS prefix (`gs://bucket/path/`) the entrypoint mirrors (rsync) into `OUTPUT_PATH/<project>/dataset/`. Used for offline-precomputed files (e.g. embedding pickles). |
| `PROJECT_NAME` | auto-detect from CLI `--project` | Forces the dataset target subdir used by `entrypoint.sh` (default: the package before `.config.` in `--project`). `EXPERIMENT_NAME` is a legacy alias. |
| `LUIGI_CONFIG_PATH` | auto (package's `luigi.cfg`) | Path to Luigi configuration; set by `cli.py` |
| `PYTHONPATH` | — | Must include the project root so `examples.*` (or your own project package) resolves |
| `PYTHONHASHSEED` | set at runtime to `seed` | Reproducibility (`training.py::seed_everything`) |

Read by the build / deploy tooling (`Makefile`, auto-loaded from `.env`):

| Variable | Default | Sensitivity | Usage |
|---|---|---|---|
| `GCP_PROJECT` | empty | **sensitive** | Artifact Registry / Cloud Build / Vertex AI project ID |
| `GCP_REGION` | `us-central1` | neutral | GCP region |
| `GCS_BUCKET` | empty | **sensitive** | Vertex AI artifact bucket (`gs://...`) |
| `IMAGE_NAME` / `IMAGE_TAG` | `mars-rec` / `latest` | neutral | Docker image identity |

`.env` is gitignored. Use `.env.example` as the template and never
commit `.env`, service-account JSON keys, `*.pem`, or `*.key`. The
GCP-touching `make` targets (`docker-push`, `docker-pull`,
`gcp-create-repo`, `gcp-submit`, `vertex-run`) error out cleanly when
`GCP_PROJECT` / `GCS_BUCKET` are missing.

## Quick tests to validate changes

```bash
# 1. Import test (checks for broken imports)
python -c "from mars_rec.simulation.training import SupervisedModelTraining; print('OK')"
python -c "from mars_rec.inference import run_inference, run_evaluate; print('OK')"

# 2. Unit tests (CI runs `make test-ci`: same suite + branch-coverage floor)
make test
make test-ci

# 3. End-to-end test with MovieLens 100K slate sample (small + fast)
PYTHONPATH=src:. python examples/movielens100k_slate/run_training.py

# 4. CLI test
mars-rec run supervised \
  --project examples.movielens100k_slate.config.movielens100k_slate \
  --recommender-module-class examples.movielens100k_slate.model.ML100KSlateRanker \
  --recommender-extra-params '{"n_factors": 32, "genre_dim": 19, "hidden_size": 64}' \
  --loss-function bce_logits --negative-proportion 0.5 --epochs 2 --device cpu

# 5. Query command tests
mars-rec list
mars-rec results --project movielens100k_slate --sort-by val_loss

# 6. Inference / re-evaluation
mars-rec predict  --task-id <ID> --data data/test.parquet --output result.json
mars-rec evaluate --task-id <ID> --ranking-metrics --split test

# 7. Architecture diagram
# Each ``run supervised`` already drops ``architecture.svg`` (trace=True)
# next to the checkpoint. The CLI lets you regenerate or inspect:
mars-rec visualize --task-id <ID>                       # SVG (default), trace on
mars-rec visualize --task-id <ID> --no-trace             # static introspection only
mars-rec visualize --task-id <ID> --output /tmp/arch.svg
mars-rec visualize --task-id <ID> --mode summary         # text table
```

## Common errors and solutions

| Error | Cause | Solution |
|---|---|---|
| `RuntimeError: target size != input size` | `forward()` returns wrong shape for the chosen loss | BCE/MSE → `[batch]` (use `.squeeze(-1)`); CE → `[batch, n_items]` |
| `ModuleNotFoundError: No module named 'examples'` (or your project package) | PYTHONPATH doesn't include project root | Run from the project root or `PYTHONPATH=. python ...` |
| `Luigi task already DONE` | A finished run with the same parameters exists (`_SUCCESS` marker) | Use `--no-cache`, delete the output directory, or change a parameter (e.g. `--observation`) |
| `KeyError` in index_mapping | INDEXABLE column not found in DataFrame | Verify ProjectConfig column names match the CSV |
| `PermissionError` on cleanup | File locked by a previous process | Manually delete the output directory |
| `issubclass() arg 1 must be a class` | `load_attr` returned an instance instead of a class | `recommender_module_class` must point to the CLASS, not an instance |
| `FileNotFoundError: weights*.ckpt` in `predict`/`evaluate` | Task dir lacks a Lightning checkpoint | Re-train, or pass the correct `--task-id` / `--model-dir` |
| `metadata.json` missing in `results` | Run was trained without `--ranking-metrics` and never re-evaluated | Run `mars-rec evaluate --task-id <ID> --ranking-metrics` |
