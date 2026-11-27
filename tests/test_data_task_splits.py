"""Split, filter, balance and download behaviour of ``mars_rec.data.task``.

``test_data_task.py`` covers the ``run()`` contract (three CSVs, sizes,
``transform_data_frame`` hooks). This module covers how rows are
*divided*, which silently changes every downstream metric when it
breaks:

- each ``dataset_split_method`` (holdout / column / time / k_fold) and
  ``test_split_type`` (random / time);
- ``eq`` / ``neq`` / ``isin`` filters and the ``output()`` naming;
- determinism of the generated ``available_items`` slates;
- over/under-sampling with proportions given as JSON (string keys) —
  the sampler itself is covered by ``test_balance_dataset.py``;
- ``BaseDownloadDataset`` delegating to ``download_file``.
"""

from __future__ import annotations

import random
from pathlib import Path

import luigi
import numpy as np
import pandas as pd
import pytest

from mars_rec.data import task as task_module
from mars_rec.data.task import BaseDownloadDataset, BasePrepareDataFrames

N_ROWS = 400
N_USERS = 40


def _raw_frame() -> pd.DataFrame:
    """400 rows, 40 users × 10 rows, 400 distinct items (> 99 so slates are
    a real sample), a strictly increasing timestamp and a 1:3 label."""
    return pd.DataFrame(
        {
            "user_id": [f"u{i % N_USERS}" for i in range(N_ROWS)],
            "item_id": [f"i{i}" for i in range(N_ROWS)],
            "ts": np.arange(N_ROWS)[::-1],  # reversed so sorting matters
            "label": [int(i % 4 == 0) for i in range(N_ROWS)],
            "country": ["br" if i % 2 else "us" for i in range(N_ROWS)],
        }
    )


class _Prep(BasePrepareDataFrames):
    """Concrete prep task; ``use_timestamp`` / ``stratify_on`` switch the
    optional hooks on, since Luigi tasks must be importable module-level
    classes configured through parameters."""

    dataset_dir_param: str = luigi.Parameter()
    use_timestamp: bool = luigi.BoolParameter(default=False)
    stratify_on: str = luigi.OptionalParameter(default=None)
    metadata_path: str = luigi.OptionalParameter(default=None)

    @property
    def dataset_dir(self):
        return self.dataset_dir_param

    @property
    def read_data_frame_path(self):
        return None

    @property
    def timestamp_property(self):
        return "ts" if self.use_timestamp else None

    @property
    def stratification_property(self):
        return self.stratify_on

    @property
    def metadata_data_frame_path(self):
        return self.metadata_path

    def read_data_frame(self):
        return _raw_frame()


@pytest.fixture(autouse=True)
def _fresh_luigi_cache():
    # Identical params would otherwise hand back a cached, stateful instance.
    luigi.task_register.Register.clear_instance_cache()
    yield


def _prep(tmp_path, **params) -> _Prep:
    params.setdefault("item_column", "item_id")
    return _Prep(dataset_dir_param=str(tmp_path), **params)


def _read_outputs(t: _Prep):
    return [pd.read_csv(target.path) for target in t.output()[:3]]


# ---------------------------------------------------------------------------
# read_data_frame / filters / output naming
# ---------------------------------------------------------------------------


class _CsvPrep(BasePrepareDataFrames):
    csv_path: str = luigi.Parameter()
    dataset_dir_param: str = luigi.Parameter()

    @property
    def dataset_dir(self):
        return self.dataset_dir_param

    @property
    def read_data_frame_path(self):
        return self.csv_path


class TestReadAndFilter:
    def test_default_read_data_frame_reads_csv(self, tmp_path):
        csv = tmp_path / "raw.csv"
        _raw_frame().to_csv(csv, index=False)
        t = _CsvPrep(csv_path=str(csv), dataset_dir_param=str(tmp_path / "out"))
        pd.testing.assert_frame_equal(t.read_data_frame(), _raw_frame())

    def test_eq_neq_isin_filters_apply_before_split(self, tmp_path):
        t = _prep(
            tmp_path,
            eq_filters={"country": "br"},
            neq_filters={"label": 1},
            isin_filters={"user_id": ["u1", "u3", "u5"]},
            dataset_split_method="holdout",
        )
        t.run()
        rows = pd.concat(_read_outputs(t))

        expected = _raw_frame()
        expected = expected[
            (expected.country == "br") & (expected.label != 1) & expected.user_id.isin(["u1", "u3", "u5"])
        ]
        assert sorted(rows["item_id"]) == sorted(expected["item_id"])

    def test_metadata_path_adds_fourth_output(self, tmp_path):
        t = _prep(tmp_path, metadata_path=str(tmp_path / "meta.csv"))
        outputs = t.output()
        assert len(outputs) == 4
        assert outputs[3].path == str(tmp_path / "meta.csv")

    def test_k_fold_outputs_are_named_per_fold(self, tmp_path):
        names = [Path(o.path).name for o in _prep(tmp_path, dataset_split_method="k_fold", n_splits=5).output()]
        assert names[0].startswith("train_[1of5]_")
        assert names[1].startswith("val_[1of5]_")
        assert names[2].startswith("test_0.20_")
        other = _prep(tmp_path, dataset_split_method="k_fold", n_splits=5, split_index=1).output()
        assert Path(other[0].path).name.startswith("train_[2of5]_")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


class TestSplitDataset:
    def test_zero_sizes_give_empty_splits(self, tmp_path):
        t = _prep(tmp_path, test_size=0.0, val_size=0.0)
        train, val, test = t.split_dataset(_raw_frame())
        assert len(train) == N_ROWS
        assert val.empty and test.empty
        assert list(val.columns) == list(train.columns)

    def test_time_test_split_holds_out_most_recent_rows(self, tmp_path):
        t = _prep(tmp_path, use_timestamp=True, test_split_type="time", dataset_split_method="time", test_size=0.25)
        train, val, test = t.split_dataset(_raw_frame())
        assert len(test) == N_ROWS // 4
        assert train["ts"].max() < val["ts"].min() <= val["ts"].max() < test["ts"].min()

    def test_time_split_without_timestamp_is_positional(self, tmp_path):
        t = _prep(tmp_path)
        train, test = t.time_train_test_split(_raw_frame(), test_size=0.1)
        pd.testing.assert_frame_equal(pd.concat([train, test]), _raw_frame())
        assert len(test) == N_ROWS // 10

    def test_holdout_is_seeded(self, tmp_path):
        a = _prep(tmp_path, seed=1, dataset_split_method="holdout").split_dataset(_raw_frame())
        b = _prep(tmp_path / "b", seed=1, dataset_split_method="holdout").split_dataset(_raw_frame())
        c = _prep(tmp_path / "c", seed=2, dataset_split_method="holdout").split_dataset(_raw_frame())
        for x, y in zip(a, b):
            assert list(x.index) == list(y.index)
        assert list(a[2].index) != list(c[2].index)

    def test_random_split_stratifies_on_property(self, tmp_path):
        t = _prep(tmp_path, stratify_on="label")
        train, test = t.random_train_test_split(_raw_frame(), test_size=0.2)
        assert train["label"].mean() == pytest.approx(0.25)
        assert test["label"].mean() == pytest.approx(0.25)

    def test_column_split_keeps_each_key_on_one_side(self, tmp_path):
        t = _prep(
            tmp_path, test_size=0.0, val_size=0.25, dataset_split_method="column", column_stratification="user_id"
        )
        train, val, _ = t.split_dataset(_raw_frame())
        assert set(train.user_id).isdisjoint(set(val.user_id))
        assert len(set(val.user_id)) == N_USERS // 4
        assert len(train) + len(val) == N_ROWS


class TestKFold:
    def test_works_without_stratification_property(self, tmp_path):
        """Regression: StratifiedKFold.split(df, None) raised TypeError, so
        ``dataset_split_method="k_fold"`` failed unless the subclass set
        ``stratification_property``."""
        train, val = _prep(tmp_path, n_splits=4).kfold_split(_raw_frame())
        assert (len(train), len(val)) == (300, 100)

    @pytest.mark.parametrize("stratify_on", [None, "label"])
    def test_folds_partition_the_rows(self, tmp_path, stratify_on):
        df = _raw_frame()
        vals = [
            _prep(tmp_path, n_splits=4, split_index=i, stratify_on=stratify_on).kfold_split(df)[1] for i in range(4)
        ]
        all_val = pd.concat(vals)
        assert sorted(all_val.index) == list(range(N_ROWS))  # every row exactly once
        if stratify_on:
            assert all(v["label"].mean() == pytest.approx(0.25) for v in vals)

    @pytest.mark.parametrize("split_index", [-1, 4])
    def test_rejects_out_of_range_split_index(self, tmp_path, split_index):
        with pytest.raises(ValueError, match="split_index"):
            _prep(tmp_path, n_splits=4, split_index=split_index).kfold_split(_raw_frame())

    def test_run_with_k_fold(self, tmp_path):
        t = _prep(tmp_path, dataset_split_method="k_fold", n_splits=4, test_size=0.2, val_size=0.2)
        t.run()
        train, val, test = _read_outputs(t)
        assert (len(train), len(val), len(test)) == (240, 80, 80)


# ---------------------------------------------------------------------------
# available_items slates
# ---------------------------------------------------------------------------


class TestCreateAvailableItems:
    def _slates(self, tmp_path, seed, global_seed):
        # A different global RNG state stands in for a fresh process.
        random.seed(global_seed)
        df = _raw_frame()
        _prep(tmp_path, seed=seed).create_available_items(df)
        return df["available_items"].tolist()

    def test_same_seed_gives_same_slates(self, tmp_path):
        """Regression: slates were drawn from the unseeded global ``random``,
        so two preps with identical params produced different candidate
        pools — and different training-time ranking metrics."""
        assert self._slates(tmp_path, seed=7, global_seed=1) == self._slates(tmp_path, seed=7, global_seed=2)

    def test_different_seed_changes_slates(self, tmp_path):
        assert self._slates(tmp_path, seed=7, global_seed=1) != self._slates(tmp_path, seed=8, global_seed=1)

    def test_slate_contains_positive_and_at_most_100_items(self, tmp_path):
        df = _raw_frame()
        _prep(tmp_path).create_available_items(df)
        for item, slate in zip(df["item_id"], df["available_items"]):
            assert item in slate
            assert len(slate) <= 100
            assert len(set(slate)) == len(slate)


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------


class TestBalanceDataset:
    """Only what ``test_balance_dataset.py`` does not cover: proportions as
    they arrive from the CLI / params.json, and validation resampling."""

    @pytest.mark.parametrize("strategy,expected", [("undersample", {0: 33, 1: 100}), ("oversample", {0: 300, 1: 900})])
    @pytest.mark.parametrize("keys", ["int", "str"])
    def test_proportions_accept_json_string_keys(self, tmp_path, keys, strategy, expected):
        """Regression: proportions from the CLI / params.json have string
        keys ("0", "1"); looking them up with the int label raised KeyError."""
        props = {0: 0.25, 1: 0.75} if keys == "int" else {"0": 0.25, "1": 0.75}
        t = _prep(tmp_path, sampling_strategy=strategy, balance_fields=["label"], sampling_proportions={"label": props})
        # counts 300/100. undersample: 0.25*100/0.75=33, 0.75*100/0.75=100.
        # oversample: 0.25*300/0.25=300, 0.75*300/0.25=900.
        assert t.balance_dataset(_raw_frame())["label"].value_counts().to_dict() == expected

    def test_class_without_proportion_is_left_untouched(self, tmp_path):
        """As documented on ``balance_dataset``: only classes listed in
        ``sampling_proportions`` are resampled."""
        t = _prep(
            tmp_path,
            sampling_strategy="oversample",
            balance_fields=["label"],
            sampling_proportions={"label": {"1": 1.0}},  # nothing for label 0
        )
        # Class 1 grows to max(count)=300 * 1.0 / 1.0; class 0 keeps its 300 rows.
        assert t.balance_dataset(_raw_frame())["label"].value_counts().to_dict() == {0: 300, 1: 300}

    @pytest.mark.parametrize("in_validation", [False, True])
    def test_validation_balanced_only_when_requested(self, tmp_path, in_validation):
        t = _prep(
            tmp_path,
            dataset_split_method="holdout",
            stratify_on="label",
            sampling_strategy="undersample",
            balance_fields=["label"],
            use_sampling_in_validation=in_validation,
        )
        train, val, test = t.split_dataset(_raw_frame())
        assert train["label"].mean() == pytest.approx(0.5)
        assert val["label"].mean() == pytest.approx(0.5 if in_validation else 0.25)
        assert test["label"].mean() == pytest.approx(0.25)  # test is never resampled


# ---------------------------------------------------------------------------
# BaseDownloadDataset (the download itself is covered by test_download.py)
# ---------------------------------------------------------------------------


class _Download(BaseDownloadDataset):
    target: str = luigi.Parameter()

    @property
    def url(self):
        return "https://example.invalid/data.bin"

    def output(self):
        return luigi.LocalTarget(self.target)


def test_download_task_delegates_to_download_file(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(task_module, "download_file", lambda url, path: calls.append((url, path)))
    target = tmp_path / "nested" / "data.bin"
    _Download(target=str(target)).run()
    assert calls == [("https://example.invalid/data.bin", str(target))]
