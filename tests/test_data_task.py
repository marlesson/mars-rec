"""Tests for ``mars_rec.data.task.BasePrepareDataFrames``.

The abstract Luigi task that every project's data-prep step inherits.
Tested by building a concrete subclass that reads from an in-memory
DataFrame, then exercising:
- run() produces the train/val/test CSVs Luigi declared as outputs;
- split sizes follow ``val_size`` / ``test_size`` parameters;
- ``transform_data_frame`` is invoked with the right ``data_key`` for
  each split.

The downstream framework (training.py) relies on the exact shape of
``output()`` to wire the requires() chain; if that contract drifts,
every project's training breaks.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import luigi
import pandas as pd
import pytest

from mars_rec.data.task import BasePrepareDataFrames


# A module-level concrete subclass — Luigi requires task classes to be
# importable, so we cannot define them inside a test function.
class _MiniPrep(BasePrepareDataFrames):
    """A concrete BasePrepareDataFrames whose source data is hardcoded.

    Each test sets the class-level ``dataset_dir_for_test`` before
    instantiating, since Luigi parameters must be parameters, not free
    attributes.
    """

    dataset_dir_param: str = luigi.Parameter()
    transform_log_path: str = luigi.OptionalParameter(default="")

    @property
    def dataset_dir(self):
        return self.dataset_dir_param

    @property
    def read_data_frame_path(self):
        return None  # we override read_data_frame()

    def read_data_frame(self):
        # 30 rows × 3 columns — small but enough to split.
        return pd.DataFrame({
            "user_id": list(range(30)),
            "item_id": list(range(100, 130)),
            "label": [i % 2 for i in range(30)],
        })

    def transform_data_frame(self, df, data_key):
        # Record the data_key per call for the test that inspects it.
        if self.transform_log_path:
            with open(self.transform_log_path, "a") as f:
                f.write(f"{data_key}|{len(df)}\n")
        return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasePrepareDataFramesRun:
    def test_run_creates_train_val_test_outputs(self, tmp_path):
        out_dir = tmp_path / "ds"
        task = _MiniPrep(
            dataset_dir_param=str(out_dir),
            test_size=0.2, val_size=0.2,
            dataset_split_method="holdout",
            test_split_type="random",
            item_column="item_id",
            seed=42,
        )
        task.run()

        train_t, val_t, test_t = task.output()
        assert os.path.exists(train_t.path)
        assert os.path.exists(val_t.path)
        assert os.path.exists(test_t.path)

    def test_split_sizes_obey_parameters(self, tmp_path):
        """30 rows, test_size=0.2 → 6 in test; remaining 24 split val_size=0.25
        → 6 val, 18 train. The numbers must match exactly."""
        out_dir = tmp_path / "ds"
        task = _MiniPrep(
            dataset_dir_param=str(out_dir),
            test_size=0.2, val_size=0.25,
            dataset_split_method="holdout",
            test_split_type="random",
            item_column="item_id",
            seed=0,
        )
        task.run()

        train_t, val_t, test_t = task.output()
        train_df = pd.read_csv(train_t.path)
        val_df = pd.read_csv(val_t.path)
        test_df = pd.read_csv(test_t.path)

        assert len(train_df) + len(val_df) + len(test_df) == 30
        assert len(test_df) == 6
        assert len(val_df) == 6
        assert len(train_df) == 18

    def test_transform_called_once_per_split_with_correct_data_key(self, tmp_path):
        """run() calls transform_data_frame three times — TRAIN_DATA,
        VALIDATION_DATA, TEST_GENERATOR — in that order."""
        out_dir = tmp_path / "ds"
        log = tmp_path / "transform_log.txt"
        log.touch()  # OptionalParameter("") would skip the log
        task = _MiniPrep(
            dataset_dir_param=str(out_dir),
            test_size=0.2, val_size=0.2,
            dataset_split_method="holdout",
            test_split_type="random",
            item_column="item_id",
            transform_log_path=str(log),
            seed=0,
        )
        task.run()

        lines = [ln.strip() for ln in log.read_text().splitlines() if ln.strip()]
        keys = [ln.split("|")[0] for ln in lines]
        # The three documented constants on BasePrepareDataFrames.
        assert keys == [
            BasePrepareDataFrames.TRAIN_DATA,
            BasePrepareDataFrames.VALIDATION_DATA,
            BasePrepareDataFrames.TEST_GENERATOR,
        ]


class TestSplitHelpers:
    def test_random_train_test_split_respects_test_size(self, tmp_path):
        task = _MiniPrep(
            dataset_dir_param=str(tmp_path),
            item_column="item_id",
            seed=0,
        )
        df = pd.DataFrame({
            "user_id": list(range(20)),
            "item_id": list(range(100, 120)),
            "label": [i % 2 for i in range(20)],
        })
        train_df, test_df = task.random_train_test_split(df, test_size=0.25)
        assert len(test_df) == 5
        assert len(train_df) == 15
        # No row appears in both splits.
        common = set(train_df["user_id"]).intersection(set(test_df["user_id"]))
        assert not common

    def test_time_train_test_split_keeps_order(self, tmp_path):
        """time split has no timestamp_property here → sorts by index, then
        splits by row position. Front rows go to train, tail to test."""
        task = _MiniPrep(
            dataset_dir_param=str(tmp_path),
            item_column="item_id",
            seed=0,
        )
        df = pd.DataFrame({"user_id": list(range(10)), "label": [0] * 10})
        train_df, test_df = task.time_train_test_split(df, test_size=0.3)
        # 70/30 split on 10 rows → 7 train + 3 test.
        assert len(train_df) == 7
        assert len(test_df) == 3
        # First train row is the original first row; first test row follows it.
        assert train_df.iloc[0]["user_id"] == 0
        assert test_df.iloc[0]["user_id"] == 7


class TestCreateAvailableItems:
    def test_adds_column_when_missing(self, tmp_path):
        task = _MiniPrep(
            dataset_dir_param=str(tmp_path),
            item_column="item_id",
            available_items_column_name="avail",
            seed=0,
        )
        df = pd.DataFrame({
            "item_id": [1, 2, 3, 4, 5],
        })
        task.create_available_items(df)
        assert "avail" in df.columns
        # Each row's available_items must include the row's own item id.
        for _, row in df.iterrows():
            assert row["item_id"] in row["avail"]

    def test_preserves_column_when_already_present(self, tmp_path):
        task = _MiniPrep(
            dataset_dir_param=str(tmp_path),
            item_column="item_id",
            available_items_column_name="avail",
            seed=0,
        )
        df = pd.DataFrame({
            "item_id": [1, 2],
            "avail": [[1, 2, 3], [4, 5]],
        })
        original_avail = df["avail"].tolist()
        task.create_available_items(df)
        # Column kept as-is (no modification).
        assert df["avail"].tolist() == original_avail
