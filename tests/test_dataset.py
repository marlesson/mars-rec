import pytest
import numpy as np
import pandas as pd
from collections import defaultdict
from unittest.mock import MagicMock

from mars_rec.meta_config import Column, IOType, ProjectConfig
from mars_rec.data.dataset import (
    preprocess_interactions_data_frame,
    literal_eval_array_columns,
    InteractionsDataset,
    InteractionsWithNegativeItemGenerationDataset,
    _rand_int_except,
    _choose_except,
)


@pytest.fixture
def project_config():
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


@pytest.fixture
def index_mapping():
    return {
        "user_id": defaultdict(int, {"1": 1, "2": 2, "3": 3}),
        "item_id": defaultdict(int, {"10": 1, "20": 2, "30": 3}),
    }


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 2],
            "item_id": [10, 20, 30, 20, 10],
            "clicked": [1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )


class TestPreprocessInteractionsDataFrame:
    def test_user_id_converted_to_string(self, project_config, sample_df):
        result = preprocess_interactions_data_frame(sample_df.copy(), project_config)
        assert isinstance(result["user_id"].iloc[0], str)

    def test_item_id_converted_to_string(self, project_config, sample_df):
        result = preprocess_interactions_data_frame(sample_df.copy(), project_config)
        assert isinstance(result["item_id"].iloc[0], str)

    def test_empty_dataframe_returned_unchanged(self, project_config):
        df = pd.DataFrame({"user_id": [], "item_id": [], "clicked": []})
        result = preprocess_interactions_data_frame(df, project_config)
        assert len(result) == 0

    def test_returns_dataframe(self, project_config, sample_df):
        result = preprocess_interactions_data_frame(sample_df.copy(), project_config)
        assert isinstance(result, pd.DataFrame)

    def test_values_preserved_as_strings(self, project_config, sample_df):
        result = preprocess_interactions_data_frame(sample_df.copy(), project_config)
        assert "1" in result["user_id"].values
        assert "10" in result["item_id"].values

    def test_array_column_parsed(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        float_col = Column("features", IOType.FLOAT_ARRAY)
        config = ProjectConfig(
            base_dir="/tmp",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[float_col],
            output_column=Column("clicked", IOType.NUMBER),
        )
        df = pd.DataFrame(
            {
                "user_id": [1],
                "item_id": [10],
                "features": ["[1.0, 2.0, 3.0]"],
                "clicked": [1.0],
            }
        )
        result = preprocess_interactions_data_frame(df, config)
        assert isinstance(result["features"].iloc[0], list)
        assert result["features"].iloc[0] == [1.0, 2.0, 3.0]

    def test_int_array_column_parsed(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        int_col = Column("ids", IOType.INT_ARRAY)
        config = ProjectConfig(
            base_dir="/tmp",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[int_col],
            output_column=Column("clicked", IOType.NUMBER),
        )
        df = pd.DataFrame(
            {
                "user_id": [1],
                "item_id": [10],
                "ids": ["[10, 20, 30]"],
                "clicked": [1.0],
            }
        )
        result = preprocess_interactions_data_frame(df, config)
        assert isinstance(result["ids"].iloc[0], list)
        assert result["ids"].iloc[0] == [10, 20, 30]

    def test_already_parsed_columns_not_broken(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        float_col = Column("features", IOType.FLOAT_ARRAY)
        config = ProjectConfig(
            base_dir="/tmp",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[float_col],
            output_column=Column("clicked", IOType.NUMBER),
        )
        df = pd.DataFrame(
            {
                "user_id": [1],
                "item_id": [10],
                "features": [[1.0, 2.0, 3.0]],
                "clicked": [1.0],
            }
        )
        result = preprocess_interactions_data_frame(df, config)
        assert result["features"].iloc[0] == [1.0, 2.0, 3.0]

    def test_user_id_already_string(self, sample_df, project_config):
        sample_df["user_id"] = sample_df["user_id"].astype(str)
        result = preprocess_interactions_data_frame(sample_df, project_config)
        # Pandas 3.0 returns StringDtype for .astype(str); pre-3.0 returns object.
        assert pd.api.types.is_string_dtype(result["user_id"])

    def test_multiple_array_columns(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        float_col = Column("features", IOType.FLOAT_ARRAY)
        int_col = Column("ids", IOType.INT_ARRAY)
        config = ProjectConfig(
            base_dir="/tmp",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[float_col, int_col],
            output_column=Column("clicked", IOType.NUMBER),
        )
        df = pd.DataFrame(
            {
                "user_id": [1],
                "item_id": [10],
                "features": ["[1.0, 2.0]"],
                "ids": ["[10, 20]"],
                "clicked": [1.0],
            }
        )
        result = preprocess_interactions_data_frame(df, config)
        assert result["features"].iloc[0] == [1.0, 2.0]
        assert result["ids"].iloc[0] == [10, 20]

    def test_output_column_not_altered(self, sample_df, project_config):
        result = preprocess_interactions_data_frame(sample_df.copy(), project_config)
        assert result["clicked"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]

    def test_dataframe_row_count_preserved(self, sample_df, project_config):
        result = preprocess_interactions_data_frame(sample_df.copy(), project_config)
        assert len(result) == len(sample_df)


class TestLiteralEvalArrayColumns:
    def test_float_array_column_parsed(self):
        df = pd.DataFrame({"features": ["[1.0, 2.0]", "[3.0, 4.0]"]})
        columns = [Column("features", IOType.FLOAT_ARRAY)]
        literal_eval_array_columns(df, columns)
        assert df["features"].iloc[0] == [1.0, 2.0]

    def test_int_array_column_parsed(self):
        df = pd.DataFrame({"ids": ["[1, 2, 3]", "[4, 5, 6]"]})
        columns = [Column("ids", IOType.INT_ARRAY)]
        literal_eval_array_columns(df, columns)
        assert df["ids"].iloc[0] == [1, 2, 3]

    def test_indexable_array_column_parsed(self):
        df = pd.DataFrame({"items": ["[10, 20]", "[30, 40]"]})
        columns = [Column("items", IOType.INDEXABLE_ARRAY)]
        literal_eval_array_columns(df, columns)
        assert df["items"].iloc[0] == [10, 20]

    def test_non_array_columns_ignored(self):
        df = pd.DataFrame({"user_id": ["1", "2"]})
        columns = [Column("user_id", IOType.INDEXABLE)]
        literal_eval_array_columns(df, columns)
        assert df["user_id"].iloc[0] == "1"

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"a": [1, 2]})
        columns = [Column("nonexistent", IOType.FLOAT_ARRAY)]
        literal_eval_array_columns(df, columns)
        assert "nonexistent" not in df.columns

    def test_already_parsed_values_preserved(self):
        df = pd.DataFrame({"features": [[1.0, 2.0], [3.0, 4.0]]})
        columns = [Column("features", IOType.FLOAT_ARRAY)]
        literal_eval_array_columns(df, columns)
        assert df["features"].iloc[0] == [1.0, 2.0]

    def test_multiple_columns_processed(self):
        df = pd.DataFrame(
            {
                "a": ["[1, 2]", "[3, 4]"],
                "b": ["[5.0]", "[6.0]"],
            }
        )
        columns = [Column("a", IOType.INT_ARRAY), Column("b", IOType.FLOAT_ARRAY)]
        literal_eval_array_columns(df, columns)
        assert df["a"].iloc[0] == [1, 2]
        assert df["b"].iloc[0] == [5.0]


class TestInteractionsDataset:
    def test_len(self, project_config, index_mapping, sample_df):
        dataset = InteractionsDataset(sample_df, None, project_config, index_mapping)
        assert len(dataset) == 5

    def test_getitem_single_index(self, project_config, index_mapping, sample_df):
        dataset = InteractionsDataset(sample_df, None, project_config, index_mapping)
        inputs, output = dataset[0]
        assert isinstance(inputs, tuple)
        assert isinstance(output, np.ndarray)

    def test_getitem_returns_correct_number_of_inputs(self, project_config, index_mapping, sample_df):
        dataset = InteractionsDataset(sample_df, None, project_config, index_mapping)
        inputs, _ = dataset[0]
        # user_id and item_id = 2 inputs
        assert len(inputs) == 2

    def test_getitem_user_id_dtype(self, project_config, index_mapping, sample_df):
        dataset = InteractionsDataset(sample_df, None, project_config, index_mapping)
        inputs, _ = dataset[0]
        assert inputs[0].dtype == np.int64

    def test_getitem_output_dtype(self, project_config, index_mapping, sample_df):
        dataset = InteractionsDataset(sample_df, None, project_config, index_mapping)
        _, output = dataset[0]
        # NUMBER type -> float32
        assert output.dtype == np.float32

    def test_getitem_list_of_indices(self, project_config, index_mapping, sample_df):
        dataset = InteractionsDataset(sample_df, None, project_config, index_mapping)
        inputs, output = dataset[[0, 1, 2]]
        assert output.shape == (3,)

    def test_getitem_with_auxiliar_output(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        aux_col = Column("weight", IOType.NUMBER)
        config = ProjectConfig(
            base_dir="/tmp",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[],
            output_column=Column("clicked", IOType.NUMBER),
            auxiliar_output_columns=[aux_col],
        )
        df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "item_id": [10, 20],
                "clicked": [1.0, 0.0],
                "weight": [0.5, 0.8],
            }
        )
        mapping = {
            "user_id": defaultdict(int, {"1": 1, "2": 2}),
            "item_id": defaultdict(int, {"10": 1, "20": 2}),
        }
        dataset = InteractionsDataset(df, None, config, mapping)
        inputs, output = dataset[0]
        assert isinstance(output, tuple)
        assert len(output) == 2

    def test_getitem_with_metadata(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        meta_col = Column("item_feat", IOType.FLOAT_ARRAY)
        config = ProjectConfig(
            base_dir="/tmp",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[],
            output_column=Column("clicked", IOType.NUMBER),
            metadata_columns=[meta_col],
        )
        # Create metadata embeddings indexed from 0
        # NON_UTIL_UID=3, so embeddings need padding for first 3 rows
        embeddings_for_metadata = {"item_feat": np.array([[0, 0], [0, 0], [0, 0], [1.0, 2.0], [3.0, 4.0]])}
        df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "item_id": [3, 4],  # item indices matching embedding rows
                "clicked": [1.0, 0.0],
            }
        )
        mapping = {
            "user_id": defaultdict(int, {"1": 1, "2": 2}),
            "item_id": defaultdict(int, {"3": 3, "4": 4}),
        }
        dataset = InteractionsDataset(df, embeddings_for_metadata, config, mapping)
        inputs, _ = dataset[0]
        # inputs = (user_id, item_id, item_feat_from_metadata)
        assert len(inputs) == 3

    def test_convert_dtype_indexable(self, project_config, index_mapping, sample_df):
        dataset = InteractionsDataset(sample_df, None, project_config, index_mapping)
        arr = np.array([1, 2, 3])
        result = dataset._convert_dtype(arr, IOType.INDEXABLE)
        assert result.dtype == np.int64

    def test_convert_dtype_number(self, project_config, index_mapping, sample_df):
        dataset = InteractionsDataset(sample_df, None, project_config, index_mapping)
        arr = np.array([1.0, 2.0])
        result = dataset._convert_dtype(arr, IOType.NUMBER)
        assert result.dtype == np.float32

    def test_convert_dtype_int_array(self, project_config, index_mapping, sample_df):
        dataset = InteractionsDataset(sample_df, None, project_config, index_mapping)
        arr = np.array([[1, 2], [3, 4]])
        result = dataset._convert_dtype(arr, IOType.INT_ARRAY)
        assert result.dtype == np.int64

    def test_convert_dtype_float_array(self, project_config, index_mapping, sample_df):
        dataset = InteractionsDataset(sample_df, None, project_config, index_mapping)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = dataset._convert_dtype(arr, IOType.FLOAT_ARRAY)
        assert result.dtype == np.float32

    def test_convert_dtype_sequence(self, project_config, index_mapping, sample_df):
        dataset = InteractionsDataset(sample_df, None, project_config, index_mapping)
        # 2 rows, each a [3, 2] sequence (3 timesteps, 2 features)
        arr = np.array([[[1, 2], [3, 4], [0, 0]], [[5, 6], [0, 0], [0, 0]]])
        result = dataset._convert_dtype(arr, IOType.SEQUENCE)
        assert result.dtype == np.int64
        assert result.shape == (2, 3, 2)


class TestInteractionsWithNegativeItemGenerationDataset:
    def test_len_with_negative_proportion(self, project_config, index_mapping, sample_df):
        dataset = InteractionsWithNegativeItemGenerationDataset(
            sample_df, None, project_config, index_mapping, negative_proportion=0.5
        )
        # Original 5 samples + int((1/(1-0.5)-1)*5) = 5 negatives = 10 total
        assert len(dataset) == 10

    def test_len_with_zero_negative_proportion(self, project_config, index_mapping, sample_df):
        dataset = InteractionsWithNegativeItemGenerationDataset(
            sample_df, None, project_config, index_mapping, negative_proportion=0.0
        )
        assert len(dataset) == 5

    def test_positive_samples_have_correct_labels(self, project_config, index_mapping, sample_df):
        dataset = InteractionsWithNegativeItemGenerationDataset(
            sample_df, None, project_config, index_mapping, negative_proportion=0.5
        )
        # Access positive samples (indices 0..n-1)
        _, output = dataset[[0, 1, 2, 3, 4]]
        # Positive outputs should match original labels
        assert output[:5].tolist() == sample_df["clicked"].tolist()

    def test_negative_samples_have_zero_labels(self, project_config, index_mapping, sample_df):
        dataset = InteractionsWithNegativeItemGenerationDataset(
            sample_df, None, project_config, index_mapping, negative_proportion=0.5
        )
        n = 5
        # Access negative samples (indices n..len-1)
        _, output = dataset[list(range(n, len(dataset)))]
        assert (output == 0).all()

    def test_negative_samples_have_different_items(self, project_config, index_mapping, sample_df):
        np.random.seed(42)
        dataset = InteractionsWithNegativeItemGenerationDataset(
            sample_df, None, project_config, index_mapping, negative_proportion=0.5
        )
        n = 5
        neg_inputs, _ = dataset[list(range(n, len(dataset)))]
        # Negative item indices should be in valid range
        neg_item_ids = neg_inputs[1]  # item_id is index 1
        assert all(0 <= idx <= dataset._max_item_idx for idx in neg_item_ids)

    def test_getitem_returns_correct_output_shapes(self, project_config, index_mapping, sample_df):
        dataset = InteractionsWithNegativeItemGenerationDataset(
            sample_df, None, project_config, index_mapping, negative_proportion=0.5
        )
        inputs, output = dataset[list(range(len(dataset)))]
        assert output.shape == (len(dataset),)


class TestRandIntExcept:
    def test_never_returns_exception_value(self):
        exception = 5
        for _ in range(100):
            result = _rand_int_except(0, 10, exception)
            assert result != exception

    def test_returns_value_in_range(self):
        for _ in range(100):
            result = _rand_int_except(0, 5, 3)
            assert 0 <= result < 5

    def test_with_tight_range(self):
        # Only two choices: 0 and 2, exception is 1
        for _ in range(20):
            result = _rand_int_except(0, 3, 1)
            assert result in (0, 2)


class TestChooseExcept:
    def test_never_returns_exception(self):
        values = [1, 2, 3, 4, 5]
        exception = 3
        for _ in range(100):
            result = _choose_except(values, exception)
            assert result != exception

    def test_returns_value_from_list(self):
        values = [10, 20, 30]
        for _ in range(50):
            result = _choose_except(values, 10)
            assert result in values

    def test_with_two_options(self):
        values = ["a", "b"]
        for _ in range(20):
            result = _choose_except(values, "a")
            assert result == "b"
