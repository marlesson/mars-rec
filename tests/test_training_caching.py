"""Caching behaviour of ``SupervisedModelTraining`` that the eval loop relies on."""

from collections import defaultdict

import pandas as pd
import pytest


def _task(**overrides):
    from mars_rec.simulation.training import SupervisedModelTraining

    params = dict(
        project="tests._fixtures.dummy_project.dummy_project_config",
        recommender_module_class="mars_rec.model.base_model.LogisticRegression",
        device="cpu",
        observation="caching-test",
    )
    params.update(overrides)
    return SupervisedModelTraining(**params)


class TestReverseIndexMapping:
    def test_built_once_per_mapping_object(self):
        task = _task()
        task._index_mapping = {"item_id": defaultdict(int, {"a": 3, "b": 4})}
        first = task.reverse_index_mapping
        assert first["item_id"] == {3: "a", 4: "b", 0: 0}
        assert task.reverse_index_mapping is first

    def test_rebuilt_when_mapping_is_replaced(self):
        task = _task()
        task._index_mapping = {"item_id": defaultdict(int, {"a": 3})}
        first = task.reverse_index_mapping
        task._index_mapping = {"item_id": defaultdict(int, {"z": 3})}
        assert task.reverse_index_mapping is not first
        assert task.reverse_index_mapping["item_id"][3] == "z"


class TestCacheCleanup:
    def test_keep_leaves_named_attributes(self):
        task = _task()
        for attr in task.cache_attrs:
            setattr(task, attr, object())
        task.cache_cleanup(keep=("_val_data_frame", "_val_dataset"))
        remaining = {a for a in task.cache_attrs if hasattr(task, a)}
        assert remaining == {"_val_data_frame", "_val_dataset"}

    def test_default_drops_everything(self):
        task = _task()
        for attr in task.cache_attrs:
            setattr(task, attr, object())
        task.cache_cleanup()
        assert not any(hasattr(task, a) for a in task.cache_attrs)


class TestDataFrameInteractions:
    def test_reads_only_distinct_user_item_pairs(self, tmp_path, monkeypatch):
        from mars_rec.simulation.training import SupervisedModelTraining

        task = _task()
        cfg = task.project_config
        user, item = cfg.user_column.name, cfg.item_column.name
        train = pd.DataFrame({user: [1, 1, 2], item: [10, 10, 20], "extra": ["[1]", "[2]", "[3]"]})
        val = pd.DataFrame({user: [2, 3], item: [20, 30], "extra": ["[4]", "[5]"]})
        train.to_csv(tmp_path / "train.csv", index=False)
        val.to_csv(tmp_path / "val.csv", index=False)
        monkeypatch.setattr(SupervisedModelTraining, "train_data_frame_path", str(tmp_path / "train.csv"))
        monkeypatch.setattr(SupervisedModelTraining, "val_data_frame_path", str(tmp_path / "val.csv"))

        pairs = task.get_data_frame_interactions()

        assert list(pairs.columns) == [user, item]
        assert sorted(map(tuple, pairs.values.tolist())) == [(1, 10), (2, 20), (3, 30)]


@pytest.fixture(autouse=True)
def _clear_luigi_cache():
    import luigi

    luigi.task_register.Register.clear_instance_cache()
    yield
    luigi.task_register.Register.clear_instance_cache()


class TestIndexMappingReuse:
    def test_complete_pickle_skips_reading_the_splits(self, tmp_path, monkeypatch):
        import pickle

        from mars_rec.meta_config import IOType
        from mars_rec.simulation.training import SupervisedModelTraining
        from mars_rec.utils.files import get_index_mapping_path

        task = _task()
        task._source_task_dir = str(tmp_path)
        cfg = task.project_config
        mapping = {
            c.name: defaultdict(int, {"x": 3})
            for c in cfg.all_columns
            if c.type in (IOType.INDEXABLE, IOType.INDEXABLE_ARRAY)
        }
        with open(get_index_mapping_path(str(tmp_path)), "wb") as f:
            pickle.dump(mapping, f)

        def _must_not_read(self):
            raise AssertionError("train/val must not be read when the mapping is complete")

        monkeypatch.setattr(SupervisedModelTraining, "get_data_frame_for_indexing", _must_not_read)

        assert set(task.index_mapping) == set(mapping)
