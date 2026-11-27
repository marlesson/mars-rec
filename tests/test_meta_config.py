import pytest
import numpy as np
from unittest.mock import MagicMock

from mars_rec.meta_config import IOType, Column, ProjectConfig, RecommenderType


class TestIOType:
    def test_indexable_dtype(self):
        assert IOType.INDEXABLE.dtype == np.int64

    def test_number_dtype(self):
        assert IOType.NUMBER.dtype == np.float32

    def test_float_array_dtype(self):
        assert IOType.FLOAT_ARRAY.dtype == np.float32

    def test_int_array_dtype(self):
        assert IOType.INT_ARRAY.dtype == np.int64

    def test_indexable_array_has_no_dtype(self):
        with pytest.raises(KeyError):
            _ = IOType.INDEXABLE_ARRAY.dtype

    def test_sequence_dtype(self):
        assert IOType.SEQUENCE.dtype == np.int64

    def test_all_enum_values_exist(self):
        assert len(IOType) == 6
        names = {e.name for e in IOType}
        assert names == {"INDEXABLE", "NUMBER", "INDEXABLE_ARRAY", "FLOAT_ARRAY", "INT_ARRAY", "SEQUENCE"}


class TestColumn:
    def test_basic_creation(self):
        col = Column("user_id", IOType.INDEXABLE)
        assert col.name == "user_id"
        assert col.type == IOType.INDEXABLE
        assert col.same_index_as is None

    def test_same_index_as(self):
        col = Column("item_feat", IOType.FLOAT_ARRAY, same_index_as="item_id")
        assert col.same_index_as == "item_id"

    def test_number_type(self):
        col = Column("price", IOType.NUMBER)
        assert col.type == IOType.NUMBER


class TestProjectConfig:
    @pytest.fixture
    def project_config(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        user_col = Column("user_id", IOType.INDEXABLE)
        item_col = Column("item_id", IOType.INDEXABLE)
        output_col = Column("clicked", IOType.NUMBER)
        return ProjectConfig(
            base_dir="/tmp/test",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=user_col,
            item_column=item_col,
            other_input_columns=[],
            output_column=output_col,
        )

    def test_basic_attributes(self, project_config):
        assert project_config.base_dir == "/tmp/test"
        assert project_config.user_column.name == "user_id"
        assert project_config.item_column.name == "item_id"
        assert project_config.output_column.name == "clicked"

    def test_input_columns_user_and_item(self, project_config):
        cols = project_config.input_columns
        names = [c.name for c in cols]
        assert "user_id" in names
        assert "item_id" in names

    def test_input_columns_order(self, project_config):
        cols = project_config.input_columns
        assert cols[0].name == "user_id"
        assert cols[1].name == "item_id"

    def test_input_columns_with_other_inputs(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        extra_col = Column("context", IOType.FLOAT_ARRAY)
        config = ProjectConfig(
            base_dir="/tmp/test",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[extra_col],
            output_column=Column("clicked", IOType.NUMBER),
        )
        names = [c.name for c in config.input_columns]
        assert "context" in names

    def test_input_columns_user_not_input(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        config = ProjectConfig(
            base_dir="/tmp/test",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[],
            output_column=Column("clicked", IOType.NUMBER),
            user_is_input=False,
        )
        names = [c.name for c in config.input_columns]
        assert "user_id" not in names
        assert "item_id" in names

    def test_input_columns_item_not_input(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        config = ProjectConfig(
            base_dir="/tmp/test",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[],
            output_column=Column("clicked", IOType.NUMBER),
            item_is_input=False,
        )
        names = [c.name for c in config.input_columns]
        assert "item_id" not in names
        assert "user_id" in names

    def test_all_columns(self, project_config):
        cols = project_config.all_columns
        names = [c.name for c in cols]
        assert "user_id" in names
        assert "item_id" in names
        assert "clicked" in names

    def test_all_columns_includes_auxiliar(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        aux_col = Column("confidence", IOType.NUMBER)
        config = ProjectConfig(
            base_dir="/tmp/test",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[],
            output_column=Column("clicked", IOType.NUMBER),
            auxiliar_output_columns=[aux_col],
        )
        names = [c.name for c in config.all_columns]
        assert "confidence" in names

    def test_get_column_by_name_found(self, project_config):
        col = project_config.get_column_by_name("user_id")
        assert col is not None
        assert col.name == "user_id"

    def test_get_column_by_name_output_col(self, project_config):
        col = project_config.get_column_by_name("clicked")
        assert col is not None
        assert col.name == "clicked"

    def test_get_column_by_name_not_found(self, project_config):
        col = project_config.get_column_by_name("nonexistent")
        assert col is None

    def test_default_recommender_type(self, project_config):
        assert project_config.recommender_type == RecommenderType.USER_BASED_COLLABORATIVE_FILTERING

    def test_metadata_columns_in_input(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        meta_col = Column("item_feature", IOType.FLOAT_ARRAY)
        config = ProjectConfig(
            base_dir="/tmp/test",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[],
            output_column=Column("clicked", IOType.NUMBER),
            metadata_columns=[meta_col],
        )
        names = [c.name for c in config.input_columns]
        assert "item_feature" in names

    def test_metadata_columns_not_in_input_when_item_not_input(self):
        prepare_task = MagicMock()
        dataset_class = MagicMock()
        meta_col = Column("item_feature", IOType.FLOAT_ARRAY)
        config = ProjectConfig(
            base_dir="/tmp/test",
            prepare_data_frames_task=prepare_task,
            dataset_class=dataset_class,
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[],
            output_column=Column("clicked", IOType.NUMBER),
            metadata_columns=[meta_col],
            item_is_input=False,
        )
        names = [c.name for c in config.input_columns]
        assert "item_feature" not in names
