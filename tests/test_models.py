import pytest
import torch
import numpy as np
from collections import defaultdict
from unittest.mock import MagicMock

from mars_rec.meta_config import Column, IOType, ProjectConfig
from mars_rec.model.abstract import RecommenderModule
from mars_rec.model.base_model import LogisticRegression


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
    user_mapping = defaultdict(int, {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5})
    item_mapping = defaultdict(int, {"10": 1, "20": 2, "30": 3, "40": 4})
    return {"user_id": user_mapping, "item_id": item_mapping}


class TestRecommenderModule:
    def test_n_users_computed_from_mapping(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        # max value in user mapping is 5, so n_users = 6
        assert model._n_users == 6

    def test_n_items_computed_from_mapping(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        # max value in item mapping is 4, so n_items = 5
        assert model._n_items == 5

    def test_index_mapping_stored(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        assert model._index_mapping is index_mapping

    def test_project_config_stored(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        assert model._project_config is project_config

    def test_recommendation_score_delegates_to_forward(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        user_ids = torch.tensor([1, 2])
        item_ids = torch.tensor([1, 2])
        score = model.recommendation_score(user_ids, item_ids)
        forward_result = model.forward(user_ids, item_ids)
        assert torch.allclose(score, forward_result)


class TestLogisticRegression:
    def test_forward_output_shape(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        user_ids = torch.tensor([1, 2, 3])
        item_ids = torch.tensor([1, 2, 3])
        output = model(user_ids, item_ids)
        assert output.shape == torch.Size([3, 1])

    def test_output_in_zero_one_range(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        user_ids = torch.tensor([1, 2, 3])
        item_ids = torch.tensor([1, 2, 3])
        output = model(user_ids, item_ids)
        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_embedding_sizes(self, project_config, index_mapping):
        n_factors = 16
        model = LogisticRegression(project_config, index_mapping, n_factors=n_factors)
        assert model.user_embeddings.embedding_dim == n_factors
        assert model.item_embeddings.embedding_dim == n_factors

    def test_user_embedding_num_embeddings(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        assert model.user_embeddings.num_embeddings == model._n_users

    def test_item_embedding_num_embeddings(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        assert model.item_embeddings.num_embeddings == model._n_items

    def test_linear_layer_shape(self, project_config, index_mapping):
        n_factors = 8
        model = LogisticRegression(project_config, index_mapping, n_factors=n_factors)
        assert model.linear.in_features == n_factors * 2
        assert model.linear.out_features == 1

    def test_different_users_produce_different_outputs(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=16)
        item_ids = torch.tensor([1, 1])
        user_ids = torch.tensor([1, 2])
        output = model(user_ids, item_ids)
        # Different users should (very likely) produce different outputs
        assert output[0].item() != output[1].item()

    def test_different_items_produce_different_outputs(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=16)
        user_ids = torch.tensor([1, 1])
        item_ids = torch.tensor([1, 2])
        output = model(user_ids, item_ids)
        assert output[0].item() != output[1].item()

    def test_forward_with_single_sample(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        user_ids = torch.tensor([1])
        item_ids = torch.tensor([1])
        output = model(user_ids, item_ids)
        assert output.shape == torch.Size([1, 1])

    def test_bias_initialized_to_point_one(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        assert model.linear.bias.data.item() == pytest.approx(0.1)

    def test_custom_weight_init(self, project_config, index_mapping):
        def custom_init(tensor):
            torch.nn.init.constant_(tensor, 0.5)

        model = LogisticRegression(project_config, index_mapping, n_factors=8, weight_init=custom_init)
        # Embedding weights should be initialized with custom init
        assert torch.all(model.user_embeddings.weight == 0.5)

    def test_batch_forward(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        batch_size = 32
        user_ids = torch.randint(1, 5, (batch_size,))
        item_ids = torch.randint(1, 4, (batch_size,))
        output = model(user_ids, item_ids)
        assert output.shape == torch.Size([batch_size, 1])

    def test_context_representation_ignored(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        user_ids = torch.tensor([1, 2])
        item_ids = torch.tensor([1, 2])
        out_no_ctx = model(user_ids, item_ids)
        out_with_ctx = model(user_ids, item_ids, context_representation=torch.randn(2, 4))
        # LogisticRegression ignores context_representation
        assert torch.allclose(out_no_ctx, out_with_ctx)

    def test_is_nn_module(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        assert isinstance(model, torch.nn.Module)

    def test_parameters_are_trainable(self, project_config, index_mapping):
        model = LogisticRegression(project_config, index_mapping, n_factors=8)
        params = list(model.parameters())
        assert len(params) > 0
        for p in params:
            assert p.requires_grad
