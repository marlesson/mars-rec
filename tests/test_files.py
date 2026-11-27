"""Tests for the artifact path helpers in :mod:`mars_rec.utils.files`."""

import os
import time

import pytest

from mars_rec.utils import files


def _touch(path, mtime=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


class TestFindCheckpoint:
    def test_none_when_missing(self, tmp_path):
        assert files.find_checkpoint(str(tmp_path)) is None

    def test_prefers_canonical_name(self, tmp_path):
        canonical = files.get_weights_path(str(tmp_path))
        _touch(canonical, mtime=time.time() - 100)
        _touch(os.path.join(files.get_model_dir(str(tmp_path)), "weights-v1.ckpt"))
        assert files.find_checkpoint(str(tmp_path)) == canonical

    def test_falls_back_to_newest_versioned_checkpoint(self, tmp_path):
        model_dir = files.get_model_dir(str(tmp_path))
        _touch(os.path.join(model_dir, "weights-v1.ckpt"), mtime=time.time() - 100)
        newest = os.path.join(model_dir, "weights-v2.ckpt")
        _touch(newest)
        assert files.find_checkpoint(str(tmp_path)) == newest

    def test_weights_path_is_a_lightning_checkpoint(self, tmp_path):
        assert files.get_weights_path(str(tmp_path)).endswith(files.CHECKPOINT_NAME + ".ckpt")


def test_every_artifact_path_lives_inside_the_task_dir(tmp_path):
    task_dir = str(tmp_path / "run")
    eval_dir = files.get_eval_dir(task_dir)
    paths = [
        files.get_params_path(task_dir),
        files.get_metadata_path(task_dir),
        files.get_weights_path(task_dir),
        files.get_index_mapping_path(task_dir),
        files.get_architecture_path(task_dir),
        files.get_summary_path(task_dir),
        files.get_history_path(task_dir),
        files.get_plot_history_path(task_dir),
        files.get_sample_train_path(task_dir),
        files.get_sample_train_keys_path(task_dir),
        files.get_sample_train_scores_path(task_dir),
        files.get_eval_results_path(eval_dir),
        files.get_test_set_predictions_path(eval_dir),
        files.get_eval_plot_path(eval_dir, "roc_curve"),
    ]
    assert len(set(paths)) == len(paths)
    for path in paths:
        assert os.path.commonpath([task_dir, path]) == task_dir


def test_params_and_metadata_roundtrip(tmp_path):
    task_dir = str(tmp_path)
    with open(files.get_params_path(task_dir), "w") as f:
        f.write('{"epochs": 3}')
    with open(files.get_metadata_path(task_dir), "w") as f:
        f.write('{"run": {"task_id": "x"}}')
    assert files.get_params(task_dir) == {"epochs": 3}
    assert files.get_metadata(task_dir)["run"]["task_id"] == "x"


class TestLayout:
    def test_artifacts_are_grouped_by_consumer(self, tmp_path):
        task_dir = str(tmp_path)

        def rel(path):
            return os.path.relpath(path, task_dir).replace(os.sep, "/")

        assert rel(files.get_params_path(task_dir)) == "params.json"
        assert rel(files.get_metadata_path(task_dir)) == "metadata.json"
        assert rel(files.get_weights_path(task_dir)) == "model/weights.ckpt"
        assert rel(files.get_index_mapping_path(task_dir)) == "model/index_mapping.pkl"
        assert rel(files.get_architecture_path(task_dir)) == "model/architecture.svg"
        assert rel(files.get_summary_path(task_dir)) == "model/summary.txt"
        assert rel(files.get_history_path(task_dir)) == "training/history.csv"
        assert rel(files.get_plot_history_path(task_dir)) == "training/plot_history.jpg"
        assert rel(files.get_training_logs_dir(task_dir)) == "training"
        assert rel(files.get_sample_train_path(task_dir)) == "training/samples/sample_train.csv"
        eval_dir = files.get_eval_dir(task_dir)
        assert rel(eval_dir) == "eval/test"
        assert rel(files.get_eval_results_path(eval_dir)) == "eval/test/metrics.json"
        assert rel(files.get_test_set_predictions_path(eval_dir)) == "eval/test/predictions.csv"
        assert rel(files.get_eval_plot_path(eval_dir, "roc_curve")) == "eval/test/plots/roc_curve.png"
        assert rel(files.get_eval_dir(task_dir, "holdout")) == "eval/holdout"

    def test_new_layout_wins_when_both_exist(self, tmp_path):
        task_dir = str(tmp_path)
        _touch(os.path.join(task_dir, "history.csv"))
        new = os.path.join(task_dir, "training", "history.csv")
        _touch(new)
        assert files.get_history_path(task_dir) == new


class TestLegacyFlatLayout:
    """Runs written before the reorganisation keep every file at the root."""

    def test_reads_fall_back_to_flat_files(self, tmp_path):
        task_dir = str(tmp_path)
        for name in ("index_mapping.pkl", "history.csv", "architecture.svg", "summary.txt", "sample_train.csv"):
            _touch(os.path.join(task_dir, name))
        assert files.get_index_mapping_path(task_dir) == os.path.join(task_dir, "index_mapping.pkl")
        assert files.get_history_path(task_dir) == os.path.join(task_dir, "history.csv")
        assert files.get_architecture_path(task_dir) == os.path.join(task_dir, "architecture.svg")
        assert files.get_summary_path(task_dir) == os.path.join(task_dir, "summary.txt")
        assert files.get_sample_train_path(task_dir) == os.path.join(task_dir, "sample_train.csv")

    def test_finds_flat_checkpoint(self, tmp_path):
        flat = os.path.join(str(tmp_path), "weights-v1.ckpt")
        _touch(flat)
        assert files.find_checkpoint(str(tmp_path)) == flat

    def test_legacy_fixture_resolves(self):
        legacy_dir = os.path.join(os.path.dirname(__file__), "_fixtures", "legacy_artifacts")
        assert files.find_checkpoint(legacy_dir) == os.path.join(legacy_dir, "weights.ckpt")
        assert files.get_index_mapping_path(legacy_dir) == os.path.join(legacy_dir, "index_mapping.pkl")


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "../x"])
def test_eval_dir_rejects_paths_and_empty_names(tmp_path, name):
    with pytest.raises(ValueError):
        files.get_eval_dir(str(tmp_path), name)


def test_ensure_parent_dir_creates_missing_dirs(tmp_path):
    path = str(tmp_path / "a" / "b" / "c.txt")
    assert files.ensure_parent_dir(path) == path
    assert os.path.isdir(tmp_path / "a" / "b")


def test_list_artifacts_is_recursive_and_relative(tmp_path):
    _touch(str(tmp_path / "params.json"))
    _touch(str(tmp_path / "model" / "weights.ckpt"))
    _touch(str(tmp_path / "eval" / "test" / "plots" / "scores.png"))
    assert files.list_artifacts(str(tmp_path)) == [
        "eval/test/plots/scores.png",
        "model/weights.ckpt",
        "params.json",
    ]
