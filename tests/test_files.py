"""Tests for the artifact path helpers in :mod:`mars_rec.utils.files`."""

import os
import time

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
