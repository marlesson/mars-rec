import pytest
import numpy as np
import pandas as pd
from collections import defaultdict
from unittest.mock import MagicMock

from mars_rec.utils.index_mapping import (
    create_index_mapping,
    create_index_mapping_from_arrays,
    map_array,
    transform_with_indexing,
)
from mars_rec.meta_config import Column, IOType, ProjectConfig


class TestCreateIndexMapping:
    def test_basic_mapping(self):
        mapping = create_index_mapping(["a", "b", "c"])
        assert "a" in mapping
        assert "b" in mapping
        assert "c" in mapping

    def test_include_unknown_default(self):
        mapping = create_index_mapping(["a", "b"])
        # Unknown values (not in mapping) should map to 0
        assert mapping["unknown_value"] == 0

    def test_include_unknown_false(self):
        mapping = create_index_mapping(["a", "b"], include_unknown=False)
        assert isinstance(mapping, dict)
        assert not isinstance(mapping, defaultdict)

    def test_indices_start_at_one_with_unknown(self):
        mapping = create_index_mapping(["x"], include_unknown=True, include_none=False)
        assert mapping["x"] == 1

    def test_indices_start_at_zero_without_unknown(self):
        values = ["x"]
        mapping = create_index_mapping(values, include_unknown=False, include_none=False)
        assert mapping["x"] == 0

    def test_sorted_order(self):
        mapping = create_index_mapping(["c", "a", "b"], include_unknown=False, include_none=False)
        assert mapping["a"] < mapping["b"] < mapping["c"]

    def test_deduplication(self):
        mapping = create_index_mapping(["a", "a", "b", "b"], include_unknown=False, include_none=False)
        assert len([k for k in mapping if k not in (None, -1)]) == 2

    def test_nan_excluded_from_values(self):
        # NaN is excluded from indexable values but gets a mapping if include_none=True
        mapping = create_index_mapping(["a", float("nan"), "b"])
        assert "a" in mapping
        assert "b" in mapping

    def test_none_mapping(self):
        mapping = create_index_mapping(["a"], include_none=True)
        assert None in mapping
        assert -1 in mapping

    def test_nan_maps_to_same_as_none(self):
        mapping = create_index_mapping(["a"], include_none=True)
        assert mapping[np.nan] == mapping[None]

    def test_string_dash_one_maps_correctly(self):
        mapping = create_index_mapping(["a"], include_none=True)
        assert mapping["-1"] == mapping[-1]

    def test_numeric_values_converted_to_string(self):
        mapping = create_index_mapping([1, 2, 3], include_unknown=False, include_none=False)
        assert "1" in mapping
        assert "2" in mapping
        assert "3" in mapping

    def test_unique_indices(self):
        mapping = create_index_mapping(["a", "b", "c", "d"], include_unknown=False, include_none=False)
        indices = list(mapping.values())
        assert len(indices) == len(set(indices))


class TestCreateIndexMappingFromArrays:
    def test_basic_arrays(self):
        arrays = [["a", "b"], ["c", "d"]]
        mapping = create_index_mapping_from_arrays(arrays)
        assert "a" in mapping
        assert "b" in mapping
        assert "c" in mapping
        assert "d" in mapping

    def test_overlapping_arrays(self):
        arrays = [["a", "b"], ["a", "c"]]
        mapping = create_index_mapping_from_arrays(arrays, include_unknown=False, include_none=False)
        # "a" appears in both but should only have one entry
        assert "a" in mapping

    def test_empty_arrays(self):
        arrays = [[], []]
        mapping = create_index_mapping_from_arrays(arrays, include_unknown=False, include_none=False)
        assert len([k for k in mapping if k not in (None, -1)]) == 0


class TestMapArray:
    def test_basic_mapping(self):
        mapping = {"1": 10, "2": 20, "3": 30}
        result = map_array([1, 2, 3], mapping)
        assert result == [10, 20, 30]

    def test_string_values(self):
        mapping = {"a": 1, "b": 2}
        result = map_array(["a", "b"], mapping)
        assert result == [1, 2]

    def test_single_element(self):
        mapping = {"x": 99}
        result = map_array(["x"], mapping)
        assert result == [99]

    def test_returns_ints(self):
        mapping = defaultdict(int, {"1": 5})
        result = map_array([1], mapping)
        assert all(isinstance(v, int) for v in result)

    def test_unknown_value_with_defaultdict(self):
        mapping = defaultdict(int, {"a": 1})
        result = map_array(["unknown"], mapping)
        assert result == [0]


class TestTransformWithIndexing:
    @pytest.fixture
    def project_config(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        return ProjectConfig(
            base_dir="/tmp/test",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[],
            output_column=Column("clicked", IOType.NUMBER),
        )

    def test_none_df_returns_none(self, project_config):
        index_mapping = {"user_id": defaultdict(int, {"1": 1, "2": 2})}
        result = transform_with_indexing(None, index_mapping, project_config)
        assert result is None

    def test_indexable_column_transformed(self, project_config):
        df = pd.DataFrame({"user_id": ["1", "2", "3"], "item_id": ["10", "20", "30"], "clicked": [1, 0, 1]})
        index_mapping = {
            "user_id": defaultdict(int, {"1": 1, "2": 2, "3": 3}),
            "item_id": defaultdict(int, {"10": 10, "20": 20, "30": 30}),
        }
        transform_with_indexing(df, index_mapping, project_config)
        assert df["user_id"].tolist() == [1, 2, 3]

    def test_indexable_array_column_transformed(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        array_col = Column("tags", IOType.INDEXABLE_ARRAY)
        config = ProjectConfig(
            base_dir="/tmp/test",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[array_col],
            output_column=Column("clicked", IOType.NUMBER),
        )
        df = pd.DataFrame({
            "user_id": ["1"],
            "item_id": ["10"],
            "tags": [["a", "b"]],
            "clicked": [1],
        })
        index_mapping = {
            "user_id": defaultdict(int, {"1": 1}),
            "item_id": defaultdict(int, {"10": 10}),
            "tags": defaultdict(int, {"a": 1, "b": 2}),
        }
        transform_with_indexing(df, index_mapping, config)
        assert df["tags"].iloc[0] == [1, 2]

    def test_column_not_in_df_is_skipped(self, project_config):
        df = pd.DataFrame({"user_id": ["1"], "item_id": ["10"], "clicked": [1]})
        index_mapping = {
            "user_id": defaultdict(int, {"1": 1}),
            "item_id": defaultdict(int, {"10": 10}),
            "nonexistent_col": defaultdict(int, {"x": 1}),  # not in df
        }
        # Should not raise
        transform_with_indexing(df, index_mapping, project_config)
        assert "nonexistent_col" not in df.columns

    def test_unknown_value_maps_to_zero(self, project_config):
        df = pd.DataFrame({"user_id": ["999"], "item_id": ["10"], "clicked": [1]})
        index_mapping = {
            "user_id": defaultdict(int, {"1": 1}),
            "item_id": defaultdict(int, {"10": 10}),
        }
        transform_with_indexing(df, index_mapping, project_config)
        assert df["user_id"].iloc[0] == 0

    def test_item_id_also_transformed(self, project_config):
        df = pd.DataFrame({"user_id": ["1"], "item_id": ["20"], "clicked": [1]})
        index_mapping = {
            "user_id": defaultdict(int, {"1": 1}),
            "item_id": defaultdict(int, {"20": 5}),
        }
        transform_with_indexing(df, index_mapping, project_config)
        assert df["item_id"].iloc[0] == 5

    def test_multiple_rows_indexable(self, project_config):
        df = pd.DataFrame({
            "user_id": ["1", "2", "1", "3"],
            "item_id": ["10", "20", "30", "10"],
            "clicked": [1, 0, 1, 0],
        })
        index_mapping = {
            "user_id": defaultdict(int, {"1": 1, "2": 2, "3": 3}),
            "item_id": defaultdict(int, {"10": 10, "20": 20, "30": 30}),
        }
        transform_with_indexing(df, index_mapping, project_config)
        assert df["user_id"].tolist() == [1, 2, 1, 3]
        assert df["item_id"].tolist() == [10, 20, 30, 10]

    def test_multiple_rows_indexable_array(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        array_col = Column("tags", IOType.INDEXABLE_ARRAY)
        config = ProjectConfig(
            base_dir="/tmp/test",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[array_col],
            output_column=Column("clicked", IOType.NUMBER),
        )
        df = pd.DataFrame({
            "user_id": ["1", "2"],
            "item_id": ["10", "20"],
            "tags": [["a", "b"], ["c", "a"]],
            "clicked": [1, 0],
        })
        index_mapping = {
            "user_id": defaultdict(int, {"1": 1, "2": 2}),
            "item_id": defaultdict(int, {"10": 10, "20": 20}),
            "tags": defaultdict(int, {"a": 1, "b": 2, "c": 3}),
        }
        transform_with_indexing(df, index_mapping, config)
        assert df["tags"].iloc[0] == [1, 2]
        assert df["tags"].iloc[1] == [3, 1]

    def test_integer_column_converted_to_str_then_mapped(self, project_config):
        df = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20], "clicked": [1, 0]})
        index_mapping = {
            "user_id": defaultdict(int, {"1": 1, "2": 2}),
            "item_id": defaultdict(int, {"10": 10, "20": 20}),
        }
        transform_with_indexing(df, index_mapping, project_config)
        assert df["user_id"].tolist() == [1, 2]
        assert df["item_id"].tolist() == [10, 20]

    def test_mutates_df_in_place(self, project_config):
        df = pd.DataFrame({"user_id": ["1"], "item_id": ["10"], "clicked": [1]})
        index_mapping = {
            "user_id": defaultdict(int, {"1": 1}),
            "item_id": defaultdict(int, {"10": 10}),
        }
        original_id = id(df)
        transform_with_indexing(df, index_mapping, project_config)
        assert id(df) == original_id
        assert df["user_id"].iloc[0] == 1

    def test_output_column_not_affected(self, project_config):
        df = pd.DataFrame({"user_id": ["1"], "item_id": ["10"], "clicked": [0.75]})
        index_mapping = {
            "user_id": defaultdict(int, {"1": 1}),
            "item_id": defaultdict(int, {"10": 10}),
        }
        transform_with_indexing(df, index_mapping, project_config)
        assert df["clicked"].iloc[0] == 0.75

    def test_empty_df(self, project_config):
        df = pd.DataFrame({"user_id": pd.Series([], dtype=str), "item_id": pd.Series([], dtype=str), "clicked": pd.Series([], dtype=float)})
        index_mapping = {
            "user_id": defaultdict(int, {"1": 1}),
            "item_id": defaultdict(int, {"10": 10}),
        }
        transform_with_indexing(df, index_mapping, project_config)
        assert len(df) == 0
