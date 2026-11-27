"""Path helpers for training artifacts.

Every training run writes its outputs under
``OUTPUT_PATH/models/<TaskClass>/results/<task_id>/``::

    params.json                  # Luigi parameters this run was launched with
    metadata.json                # Run summary: environment, history, ranking / ROC-AUC metrics
    weights.ckpt                 # Lightning checkpoint (best epoch)
    index_mapping.pkl            # Categorical → int mapping (needed for inference)
    architecture.svg             # Model architecture diagram
    summary.txt                  # Layer / parameter summary
    history.csv                  # Per-epoch metrics
    plot_history.jpg             # Training curves
    lightning_logs/metrics.csv   # Raw per-step metrics (CSVLogger)
    sample_train.csv             # 100 training rows (after indexing)
    sample_train_keys.csv        # Raw key columns of those rows (optional)
    sample_train_scores.csv      # Model logits for those rows
    test_set_predictions.csv     # Per-test-row predictions
    eval_results.json            # Metrics written by ``mars-rec evaluate``
    plot_*.png                   # Score distribution, confusion matrix, ROC and PR curves

**Every** path inside a task directory is built by a function of this
module: the rest of the framework never joins a file name onto a
``task_dir`` by hand, so the layout is defined in one place.
:data:`OUTPUT_PATH` is configurable via the ``OUTPUT_PATH`` env var (or
``AIP_MODEL_DIR`` when running on Vertex AI).
"""

import glob
import json
import os
from typing import Optional, Type


# OUTPUT_PATH resolution order:
# 1. OUTPUT_PATH env var (explicit override)
# 2. AIP_MODEL_DIR env var (set by Vertex AI Custom Training)
# 3. "output" (local default)
OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH",
    os.environ.get("AIP_MODEL_DIR", "output"),
)

#: Base name of the Lightning checkpoint (``ModelCheckpoint(filename=...)``).
CHECKPOINT_NAME = "weights"

#: Name of the evaluation produced by training on the original test split.
DEFAULT_EVAL_NAME = "test"


# ---------------------------------------------------------------------------
# Task directories
# ---------------------------------------------------------------------------


def get_task_dir(model_cls: Type, task_id: str) -> str:
    """Resolve ``task_id`` to its on-disk directory for ``model_cls``."""
    return os.path.join(OUTPUT_PATH, "models", model_cls.__name__, "results", task_id)


def get_tensorboard_logdir(task_id: str) -> str:
    """Per-run TensorBoard log directory under ``OUTPUT_PATH/tensorboard_logs/``.

    Kept outside the task directory on purpose, so a single
    ``tensorboard --logdir OUTPUT_PATH/tensorboard_logs`` compares every run.
    """
    return os.path.join(OUTPUT_PATH, "tensorboard_logs", task_id)


# ---------------------------------------------------------------------------
# Run identity and summary
# ---------------------------------------------------------------------------


def get_params_path(task_dir: str) -> str:
    """Path to ``params.json`` inside a training task directory."""
    return os.path.join(task_dir, "params.json")


def get_params(task_dir: str) -> dict:
    """Load and return the parsed ``params.json`` for a training task."""
    with open(get_params_path(task_dir), "r") as params_file:
        return json.load(params_file)


def get_metadata_path(task_dir: str) -> str:
    """Path to ``metadata.json`` (run summary + ranking / ROC-AUC metrics)."""
    return os.path.join(task_dir, "metadata.json")


def get_metadata(task_dir: str) -> dict:
    """Load and return the parsed ``metadata.json`` for a training task."""
    with open(get_metadata_path(task_dir), "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model (everything inference needs)
# ---------------------------------------------------------------------------


def get_model_dir(task_dir: str) -> str:
    """Directory holding the checkpoint (``ModelCheckpoint(dirpath=...)``)."""
    return task_dir


def get_weights_path(task_dir: str) -> str:
    """Path the Lightning checkpoint is written to (``weights.ckpt``).

    To *read* a checkpoint use :func:`find_checkpoint`, which also accepts
    the ``weights-v1.ckpt``-style names written by older versions.
    """
    return os.path.join(get_model_dir(task_dir), CHECKPOINT_NAME + ".ckpt")


def find_checkpoint(task_dir: str) -> Optional[str]:
    """Return the checkpoint of a run, or ``None`` when there is none.

    Prefers ``weights.ckpt``; falls back to the most recently modified
    ``weights*.ckpt`` (older Lightning versions appended ``-v1``, ``-v2``, …
    when re-training into an existing directory).
    """
    path = get_weights_path(task_dir)
    if os.path.isfile(path):
        return path
    candidates = glob.glob(os.path.join(get_model_dir(task_dir), CHECKPOINT_NAME + "*.ckpt"))
    return max(candidates, key=os.path.getmtime) if candidates else None


def get_index_mapping_path(task_dir: str) -> str:
    """Path to ``index_mapping.pkl`` (categorical → integer lookup)."""
    return os.path.join(task_dir, "index_mapping.pkl")


def get_architecture_path(task_dir: str) -> str:
    """Path to the model architecture diagram (``architecture.svg``)."""
    return os.path.join(task_dir, "architecture.svg")


def get_summary_path(task_dir: str) -> str:
    """Path to the layer / parameter summary (``summary.txt``)."""
    return os.path.join(task_dir, "summary.txt")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def get_history_path(task_dir: str) -> str:
    """Path to the per-epoch metrics CSV (``history.csv``)."""
    return os.path.join(task_dir, "history.csv")


def get_plot_history_path(task_dir: str) -> str:
    """Path to the training-curves plot (``plot_history.jpg``)."""
    return os.path.join(task_dir, "plot_history.jpg")


def get_training_logs_dir(task_dir: str) -> str:
    """``save_dir`` of the Lightning ``CSVLogger`` (it writes ``lightning_logs/metrics.csv`` below it)."""
    return task_dir


def get_sample_train_path(task_dir: str) -> str:
    """Path to ``sample_train.csv`` (100 training rows, after indexing)."""
    return os.path.join(task_dir, "sample_train.csv")


def get_sample_train_keys_path(task_dir: str) -> str:
    """Path to ``sample_train_keys.csv`` (raw key columns of the sampled rows)."""
    return os.path.join(task_dir, "sample_train_keys.csv")


def get_sample_train_scores_path(task_dir: str) -> str:
    """Path to ``sample_train_scores.csv`` (model logits of the sampled rows)."""
    return os.path.join(task_dir, "sample_train_scores.csv")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def get_eval_dir(task_dir: str, name: str = DEFAULT_EVAL_NAME) -> str:
    """Directory holding the artifacts of the evaluation called ``name``."""
    return task_dir


def get_eval_results_path(eval_dir: str) -> str:
    """Path to the metrics file of one evaluation (``eval_results.json``)."""
    return os.path.join(eval_dir, "eval_results.json")


def get_test_set_predictions_path(eval_dir: str) -> str:
    """Path to the per-row predictions of one evaluation (``test_set_predictions.csv``)."""
    return os.path.join(eval_dir, "test_set_predictions.csv")


def get_eval_plot_path(eval_dir: str, name: str) -> str:
    """Path to an evaluation plot, e.g. ``name="roc_curve"`` → ``plot_roc_curve.png``."""
    return os.path.join(eval_dir, f"plot_{name}.png")
