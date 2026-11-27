"""Path helpers for training artifacts.

Every training run writes its outputs under
``OUTPUT_PATH/models/<TaskClass>/results/<task_id>/``, grouped by who
consumes them::

    _SUCCESS                         # Empty marker written when the run finished (Luigi completion)
    params.json                      # Luigi parameters this run was launched with
    metadata.json                    # Run summary: environment, history, test metrics, artifact list
    model/                           # Everything inference needs
        weights.ckpt                 #   Lightning checkpoint (best epoch)
        index_mapping.pkl            #   Categorical → int mapping
        architecture.svg             #   Model architecture diagram
        summary.txt                  #   Layer / parameter summary
    training/                        # How the model was trained
        history.csv                  #   Per-epoch metrics
        plot_history.jpg             #   Training curves
        lightning_logs/metrics.csv   #   Raw per-step metrics (CSVLogger)
        samples/
            sample_train.csv         #   100 training rows (after indexing)
            sample_train_keys.csv    #   Raw key columns of those rows (optional)
            sample_train_scores.csv  #   Model logits for those rows
    eval/                            # One sub-directory per evaluation
        test/                        #   Post-training evaluation on the original test split
            metrics.json             #     Point-wise, ranking and ROC-AUC metrics
            predictions.csv          #     Per-row scores and ranking
            plots/                   #     scores, confusion_matrix, roc_curve, precision_recall_curve
        <name>/                      #   ``mars-rec evaluate --data <file> [--name <name>]``

**Every** path inside a task directory is built by a function of this
module: the rest of the framework never joins a file name onto a
``task_dir`` by hand, so the layout is defined in one place.

Runs written by older versions keep every file flat in the task
directory. The helpers that point at a file return that legacy flat
path when it exists and the new one does not, so ``predict``,
``evaluate``, ``visualize``, ``list`` and ``results`` keep working on
old runs. New files are always written to the new layout.

:data:`OUTPUT_PATH` is configurable via the ``OUTPUT_PATH`` env var (or
``AIP_MODEL_DIR`` when running on Vertex AI).
"""

import glob
import json
import os
from typing import List, Optional, Type


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


def get_success_marker_path(task_dir: str) -> str:
    """Path to the ``_SUCCESS`` marker, written as the very last step of a successful run."""
    return os.path.join(task_dir, "_SUCCESS")


def is_complete_run(task_dir: str) -> bool:
    """Whether ``task_dir`` holds a finished training run.

    A run is finished when its ``_SUCCESS`` marker exists. Runs written by
    versions that predate the marker use the flat layout (no ``model/``
    directory); they count as finished when they have both a checkpoint and
    ``metadata.json``, the last artifact those versions wrote.
    """
    if os.path.isfile(get_success_marker_path(task_dir)):
        return True
    is_legacy_layout = not os.path.isdir(get_model_dir(task_dir))
    return is_legacy_layout and os.path.isfile(get_metadata_path(task_dir)) and find_checkpoint(task_dir) is not None


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

MODEL_DIR = "model"
TRAINING_DIR = "training"
SAMPLES_DIR = os.path.join(TRAINING_DIR, "samples")
EVAL_DIR = "eval"


def _resolve(task_dir: str, relative: str) -> str:
    """Return ``task_dir/relative``, or the flat legacy path if only that one exists."""
    path = os.path.join(task_dir, relative)
    legacy = os.path.join(task_dir, os.path.basename(relative))
    if legacy != path and not os.path.exists(path) and os.path.exists(legacy):
        return legacy
    return path


def ensure_parent_dir(path: str) -> str:
    """Create the parent directory of ``path`` if needed and return ``path``."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def list_artifacts(task_dir: str) -> List[str]:
    """Sorted paths (relative to ``task_dir``, ``/``-separated) of every file in a task directory."""
    artifacts = []
    for root, _dirs, names in os.walk(task_dir):
        for name in names:
            rel = os.path.relpath(os.path.join(root, name), task_dir)
            artifacts.append(rel.replace(os.sep, "/"))
    return sorted(artifacts)


# ---------------------------------------------------------------------------
# Model (everything inference needs)
# ---------------------------------------------------------------------------


def get_model_dir(task_dir: str) -> str:
    """Directory holding the checkpoint and the other inference artifacts (``model/``)."""
    return os.path.join(task_dir, MODEL_DIR)


def get_weights_path(task_dir: str) -> str:
    """Path the Lightning checkpoint is written to (``model/weights.ckpt``).

    To *read* a checkpoint use :func:`find_checkpoint`, which also finds
    checkpoints of runs written by older versions.
    """
    return os.path.join(get_model_dir(task_dir), CHECKPOINT_NAME + ".ckpt")


def find_checkpoint(task_dir: str) -> Optional[str]:
    """Return the checkpoint of a run, or ``None`` when there is none.

    Prefers ``model/weights.ckpt``. Falls back to the most recently
    modified ``weights*.ckpt`` in ``model/`` and then in the task directory
    itself: older versions wrote checkpoints flat in the task directory and
    appended ``-v1``, ``-v2``, … when re-training into an existing one.
    """
    path = get_weights_path(task_dir)
    if os.path.isfile(path):
        return path
    for directory in (get_model_dir(task_dir), task_dir):
        candidates = glob.glob(os.path.join(directory, CHECKPOINT_NAME + "*.ckpt"))
        if candidates:
            return max(candidates, key=os.path.getmtime)
    return None


def get_index_mapping_path(task_dir: str) -> str:
    """Path to ``model/index_mapping.pkl`` (categorical → integer lookup)."""
    return _resolve(task_dir, os.path.join(MODEL_DIR, "index_mapping.pkl"))


def get_architecture_path(task_dir: str) -> str:
    """Path to the model architecture diagram (``model/architecture.svg``)."""
    return _resolve(task_dir, os.path.join(MODEL_DIR, "architecture.svg"))


def get_summary_path(task_dir: str) -> str:
    """Path to the layer / parameter summary (``model/summary.txt``)."""
    return _resolve(task_dir, os.path.join(MODEL_DIR, "summary.txt"))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def get_history_path(task_dir: str) -> str:
    """Path to the per-epoch metrics CSV (``training/history.csv``)."""
    return _resolve(task_dir, os.path.join(TRAINING_DIR, "history.csv"))


def get_plot_history_path(task_dir: str) -> str:
    """Path to the training-curves plot (``training/plot_history.jpg``)."""
    return _resolve(task_dir, os.path.join(TRAINING_DIR, "plot_history.jpg"))


def get_training_logs_dir(task_dir: str) -> str:
    """``save_dir`` of the Lightning ``CSVLogger`` (it writes ``lightning_logs/metrics.csv`` below it)."""
    return os.path.join(task_dir, TRAINING_DIR)


def get_sample_train_path(task_dir: str) -> str:
    """Path to ``training/samples/sample_train.csv`` (100 training rows, after indexing)."""
    return _resolve(task_dir, os.path.join(SAMPLES_DIR, "sample_train.csv"))


def get_sample_train_keys_path(task_dir: str) -> str:
    """Path to ``training/samples/sample_train_keys.csv`` (raw key columns of the sampled rows)."""
    return _resolve(task_dir, os.path.join(SAMPLES_DIR, "sample_train_keys.csv"))


def get_sample_train_scores_path(task_dir: str) -> str:
    """Path to ``training/samples/sample_train_scores.csv`` (model logits of the sampled rows)."""
    return _resolve(task_dir, os.path.join(SAMPLES_DIR, "sample_train_scores.csv"))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def validate_eval_name(name: str) -> str:
    """Check that ``name`` is usable as a single directory name under ``eval/``."""
    if not name or name in (".", "..") or "/" in name or os.sep in name:
        raise ValueError(f"Invalid evaluation name {name!r}: it must be a plain directory name.")
    return name


def get_eval_dir(task_dir: str, name: str = DEFAULT_EVAL_NAME) -> str:
    """Directory holding the artifacts of the evaluation called ``name`` (``eval/<name>/``)."""
    return os.path.join(task_dir, EVAL_DIR, validate_eval_name(name))


def get_eval_results_path(eval_dir: str) -> str:
    """Path to the metrics file of one evaluation (``metrics.json``)."""
    return os.path.join(eval_dir, "metrics.json")


def get_test_set_predictions_path(eval_dir: str) -> str:
    """Path to the per-row predictions of one evaluation (``predictions.csv``)."""
    return os.path.join(eval_dir, "predictions.csv")


def get_eval_plot_path(eval_dir: str, name: str) -> str:
    """Path to an evaluation plot, e.g. ``name="roc_curve"`` → ``plots/roc_curve.png``."""
    return os.path.join(eval_dir, "plots", f"{name}.png")
