"""Tests for ``mars_rec.utils.reflection.load_attr``.

Used everywhere the framework needs to dynamically resolve a dotted path
to a class (project configs, recommender modules, loss functions, etc).
Wrong-type errors must surface at load time — not after the trainer is
half built.
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from mars_rec.model.abstract import RecommenderModule
from mars_rec.model.base_model import LogisticRegression
from mars_rec.utils.reflection import get_attribute_names, load_attr


class TestLoadAttr:
    def test_loads_class_via_dotted_path(self):
        loaded = load_attr(
            "mars_rec.model.base_model.LogisticRegression", RecommenderModule,
        )
        assert loaded is LogisticRegression

    def test_loads_class_when_expected_type_is_torch_module(self):
        loaded = load_attr(
            "mars_rec.model.base_model.LogisticRegression", nn.Module,
        )
        assert loaded is LogisticRegression

    def test_class_not_subclass_of_expected_raises_value_error(self):
        # ``dict`` is a class but not a subclass of nn.Module.
        with pytest.raises(ValueError, match="should be a sub class"):
            load_attr("builtins.dict", nn.Module)

    def test_non_class_attribute_must_be_instance_of_expected(self):
        """When the resolved attribute is not a class, the function checks
        ``isinstance``. ``math.pi`` is a float, not an nn.Module → ValueError."""
        with pytest.raises(ValueError, match="should be of type"):
            load_attr("math.pi", nn.Module)

    def test_missing_module_raises_module_not_found(self):
        with pytest.raises(ModuleNotFoundError):
            load_attr("definitely_not_a_real_module_xyz.SomeClass", object)

    def test_missing_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            load_attr("mars_rec.cli.ThisSymbolDoesNotExist", object)

    def test_loaded_class_is_instantiable_with_expected_signature(self):
        """End-to-end: load LogisticRegression, then build it. This is the
        scenario inference.py uses to materialise the trained model class
        from params.json."""
        from collections import defaultdict
        from unittest.mock import MagicMock

        from mars_rec.meta_config import Column, IOType, ProjectConfig

        cls = load_attr(
            "mars_rec.model.base_model.LogisticRegression", RecommenderModule,
        )
        cfg = ProjectConfig(
            base_dir="/tmp/x",
            prepare_data_frames_task=MagicMock(),
            dataset_class=MagicMock(),
            user_column=Column("user_id", IOType.INDEXABLE),
            item_column=Column("item_id", IOType.INDEXABLE),
            other_input_columns=[],
            output_column=Column("label", IOType.NUMBER),
        )
        idx = {
            "user_id": defaultdict(int, {"u1": 3}),
            "item_id": defaultdict(int, {"i1": 3}),
        }
        instance = cls(project_config=cfg, index_mapping=idx, n_factors=4)
        assert isinstance(instance, RecommenderModule)


class TestGetAttributeNames:
    def test_returns_non_routine_attribute_names(self):
        class _C:
            x = 1
            y = 2

            def method(self):
                pass

        names = get_attribute_names(_C)
        assert "x" in names and "y" in names
        # ``method`` is routine → excluded.
        assert "method" not in names
