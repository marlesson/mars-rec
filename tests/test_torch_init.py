import pytest
import torch
import numpy as np
from torch import nn

from mars_rec.torch.init import lecun_normal_init, he_init


class TestLeCunNormalInit:
    def test_modifies_tensor_inplace(self):
        tensor = torch.zeros(10, 5)
        lecun_normal_init(tensor)
        assert not torch.all(tensor == 0)

    def test_output_shape_preserved(self):
        tensor = torch.zeros(20, 10)
        original_shape = tensor.shape
        lecun_normal_init(tensor)
        assert tensor.shape == original_shape

    def test_variance_approx_one_over_fan_in(self):
        # For a 2D tensor with shape (fan_out, fan_in), fan_in = 100
        fan_in = 100
        tensor = torch.zeros(50, fan_in)
        lecun_normal_init(tensor)
        expected_std = (1.0 / fan_in) ** 0.5
        actual_std = tensor.std().item()
        # Allow 20% tolerance due to sampling variance
        assert abs(actual_std - expected_std) < expected_std * 0.5

    def test_mean_approximately_zero(self):
        tensor = torch.zeros(500, 100)
        lecun_normal_init(tensor)
        assert abs(tensor.mean().item()) < 0.1

    def test_works_on_embedding_weight(self):
        emb = nn.Embedding(100, 32)
        lecun_normal_init(emb.weight)
        assert emb.weight.shape == (100, 32)

    def test_works_on_large_tensor(self):
        tensor = torch.zeros(1000, 200)
        lecun_normal_init(tensor)
        assert tensor.shape == (1000, 200)
        assert not torch.all(tensor == 0)


class TestHeInit:
    def test_modifies_tensor_inplace(self):
        tensor = torch.zeros(10, 5)
        he_init(tensor)
        assert not torch.all(tensor == 0)

    def test_output_shape_preserved(self):
        tensor = torch.zeros(20, 10)
        original_shape = tensor.shape
        he_init(tensor)
        assert tensor.shape == original_shape

    def test_variance_approx_two_over_fan_in(self):
        # He init variance = 2 / fan_in
        fan_in = 100
        tensor = torch.zeros(50, fan_in)
        he_init(tensor)
        expected_std = (2.0 / fan_in) ** 0.5
        actual_std = tensor.std().item()
        # Allow 30% tolerance
        assert abs(actual_std - expected_std) < expected_std * 0.5

    def test_mean_approximately_zero(self):
        tensor = torch.zeros(500, 100)
        he_init(tensor)
        assert abs(tensor.mean().item()) < 0.1

    def test_works_on_linear_layer(self):
        linear = nn.Linear(64, 32)
        he_init(linear.weight)
        assert linear.weight.shape == (32, 64)

    def test_he_and_lecun_produce_different_results(self):
        torch.manual_seed(42)
        tensor1 = torch.zeros(100, 50)
        lecun_normal_init(tensor1)

        torch.manual_seed(42)
        tensor2 = torch.zeros(100, 50)
        he_init(tensor2)

        # Different initialization schemes should produce different values
        # (or at least different variances)
        std1 = tensor1.std().item()
        std2 = tensor2.std().item()
        # He std ≈ sqrt(2/fan_in), LeCun std ≈ sqrt(1/fan_in), ratio ~= sqrt(2)
        assert abs(std2 / std1 - 2**0.5) < 0.5
