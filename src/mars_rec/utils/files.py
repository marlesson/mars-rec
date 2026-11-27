"""Path helpers for training artifacts.

Every training run writes its outputs under a stable layout::

    OUTPUT_PATH/models/<TaskClass>/results/<task_id>/
        params.json              # Luigi parameters this run was launched with
        weights.pt               # Trained model checkpoint
        history.csv              # Per-epoch metrics
        metadata.json            # Post-training ranking / ROC-AUC metrics
        test_set_predictions.csv # Per-test-row predictions
        index_mapping.pkl        # Categorical → int mapping (for inference)

The functions in this module build those paths from a ``task_dir`` or
``task_id``. :data:`OUTPUT_PATH` is configurable via the
``OUTPUT_PATH`` env var (or ``AIP_MODEL_DIR`` when running on Vertex AI).
"""
import os
import json
from typing import Type


# OUTPUT_PATH resolution order:
# 1. OUTPUT_PATH env var (explicit override)
# 2. AIP_MODEL_DIR env var (set by Vertex AI Custom Training)
# 3. "output" (local default)
OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH",
    os.environ.get("AIP_MODEL_DIR", "output"),
)


def get_params_path(task_dir: str) -> str:
    """Path to ``params.json`` inside a training task directory."""
    return os.path.join(task_dir, "params.json")


def get_params(task_dir: str) -> dict:
    """Load and return the parsed ``params.json`` for a training task."""
    with open(get_params_path(task_dir), "r") as params_file:
        return json.load(params_file)


def get_weights_path(task_dir: str) -> str:
    """Path to the PyTorch checkpoint produced by the training task."""
    return os.path.join(task_dir, "weights.pt")


def get_history_path(task_dir: str) -> str:
    """Path to the per-epoch metrics CSV (``history.csv``)."""
    return os.path.join(task_dir, "history.csv")


def get_tensorboard_logdir(task_id: str) -> str:
    """Per-run TensorBoard log directory under ``OUTPUT_PATH/tensorboard_logs/``."""
    return os.path.join(OUTPUT_PATH, "tensorboard_logs", task_id)


def get_task_dir(model_cls: Type, task_id: str):
    """Resolve ``task_id`` to its on-disk directory for ``model_cls``."""
    return os.path.join(OUTPUT_PATH, "models", model_cls.__name__, "results", task_id)


def get_test_set_predictions_path(task_dir: str) -> str:
    """Path to ``test_set_predictions.csv`` for a training task."""
    return os.path.join(task_dir, "test_set_predictions.csv")


def get_metadata_path(task_dir: str) -> str:
    """Path to ``metadata.json`` (post-training ranking / ROC-AUC metrics)."""
    return os.path.join(task_dir, "metadata.json")


def get_metadata(task_dir: str) -> dict:
    """Load and return the parsed ``metadata.json`` for a training task."""
    with open(get_metadata_path(task_dir), "r") as f:
        return json.load(f)


def get_index_mapping_path(task_dir: str) -> str:
    """Path to ``index_mapping.pkl`` (categorical → integer lookup)."""
    return os.path.join(task_dir, "index_mapping.pkl")
