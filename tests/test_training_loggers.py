"""``SupervisedModelTraining._get_loggers``: TensorBoard is an optional extra.

CSV logging always runs (it is the source of ``history.csv``). The
TensorBoard logger is only added when a backend (``tensorboard`` or
``tensorboardX``, installed via ``mars-rec[tensorboard]``) is importable.
"""

from __future__ import annotations

import importlib.util

import luigi
import pytest
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import mars_rec.simulation.training as training


@pytest.fixture
def task(tmp_path, monkeypatch):
    luigi.task_register.Register.clear_instance_cache()
    monkeypatch.chdir(tmp_path)
    return training.SupervisedModelTraining(
        project="tests._fixtures.dummy_project.dummy_project_config",
        recommender_module_class="mars_rec.model.base_model.LogisticRegression",
        recommender_extra_params={"n_factors": 4},
        device="cpu",
    )


def _fake_find_spec(available: set):
    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in ("tensorboard", "tensorboardX"):
            return real("json") if name in available else None
        return real(name, *args, **kwargs)

    return find_spec


def test_without_a_tensorboard_backend_only_csv_is_used(task, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(set()))
    assert training._tensorboard_available() is False
    loggers = task._get_loggers()
    assert [type(lg) for lg in loggers] == [CSVLogger]


@pytest.mark.parametrize("backend", ["tensorboard", "tensorboardX"])
def test_either_backend_enables_tensorboard(monkeypatch, backend):
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec({backend}))
    assert training._tensorboard_available() is True


@pytest.mark.skipif(not training._tensorboard_available(), reason="mars-rec[tensorboard] not installed")
def test_with_a_backend_tensorboard_logger_is_added(task):
    loggers = task._get_loggers()
    assert [type(lg) for lg in loggers] == [CSVLogger, TensorBoardLogger]
