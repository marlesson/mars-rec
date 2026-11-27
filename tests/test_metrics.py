import pytest
import torch
import math

from mars_rec.torch.metrics import (
    bce,
    binary_accuracy,
    precision,
    recall,
    f1_score,
    mse,
    mae,
    rmse,
    accuracy,
    top_k_accuracy,
    top_5_accuracy,
    top_10_accuracy,
    top_20_accuracy,
    ndcg_at_k,
    ndcg_at_5,
    ndcg_at_10,
    ndcg_at_20,
    mrr_at_k,
    mrr_at_10,
    METRIC_REGISTRY,
    _preprocess,
)


class TestPreprocess:
    def test_sequence_handling(self):
        y_pred = torch.tensor([0.5])
        y_true = (torch.tensor([1.0]), torch.tensor([0.0]))
        _, result = _preprocess(y_pred, y_true)
        assert torch.equal(result, torch.tensor([1.0]))

    def test_non_sequence_passthrough(self):
        y_pred = torch.tensor([0.5])
        y_true = torch.tensor([1.0])
        p, t = _preprocess(y_pred, y_true)
        assert torch.equal(p, y_pred)
        assert torch.equal(t, y_true)

    def test_sparse_to_dense(self):
        indices = torch.tensor([[0]])
        values = torch.tensor([1.0])
        y_true = torch.sparse_coo_tensor(indices, values, (1,))
        y_pred = torch.tensor([0.5])
        _, result = _preprocess(y_pred, y_true)
        assert result.layout == torch.strided


class TestMetricRegistry:
    def test_all_metrics_registered(self):
        expected = {
            "bce",
            "binary_accuracy",
            "precision",
            "recall",
            "f1_score",
            "mse",
            "mae",
            "rmse",
            "accuracy",
            "top_5_accuracy",
            "top_10_accuracy",
            "top_20_accuracy",
            "ndcg@5",
            "ndcg@10",
            "ndcg@20",
            "mrr@10",
            # Embedding similarity — used by EmbeddingRecommenderLightningModule.
            "avg_cosine_similarity",
        }
        assert set(METRIC_REGISTRY.keys()) == expected

    def test_registry_binary_functions_callable(self):
        """Binary-head metrics accept 1D ``y_pred`` (one logit per row)."""
        y_pred = torch.tensor([0.8, 0.2])
        y_true = torch.tensor([1.0, 0.0])
        binary_metric_names = {
            "bce",
            "binary_accuracy",
            "precision",
            "recall",
            "f1_score",
            "mse",
            "mae",
            "rmse",
        }
        for name in binary_metric_names:
            fn = METRIC_REGISTRY[name]
            result = fn(y_pred, y_true)
            assert isinstance(result, torch.Tensor), f"{name} did not return a tensor"

    def test_registry_multiclass_callable(self):
        """Multi-class ranking metrics should accept logits [B, C] without crashing."""
        y_pred = torch.randn(4, 10)  # [B=4, C=10]
        y_true = torch.tensor([2, 5, 0, 7])  # class indices
        multiclass_metrics = {
            "accuracy",
            "top_5_accuracy",
            "top_10_accuracy",
            "top_20_accuracy",
            "ndcg@5",
            "ndcg@10",
            "ndcg@20",
            "mrr@10",
        }
        for name in multiclass_metrics:
            fn = METRIC_REGISTRY[name]
            result = fn(y_pred, y_true)
            assert isinstance(result, torch.Tensor), f"{name} failed on multiclass"


class TestBinaryAccuracy:
    def test_perfect_predictions(self):
        y_pred = torch.tensor([0.9, 0.1, 0.8, 0.2])
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        acc = binary_accuracy(y_pred, y_true)
        assert acc.item() == pytest.approx(1.0)

    def test_all_wrong(self):
        y_pred = torch.tensor([0.1, 0.9, 0.2, 0.8])
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        acc = binary_accuracy(y_pred, y_true)
        assert acc.item() == pytest.approx(0.0)

    def test_half_correct(self):
        y_pred = torch.tensor([0.9, 0.9, 0.1, 0.1])
        y_true = torch.tensor([1.0, 0.0, 1.0, 0.0])
        acc = binary_accuracy(y_pred, y_true)
        assert acc.item() == pytest.approx(0.5)

    def test_threshold_boundary(self):
        y_pred = torch.tensor([0.5, 0.5])
        y_true = torch.tensor([1.0, 0.0])
        acc = binary_accuracy(y_pred, y_true)
        assert acc.item() == pytest.approx(0.5)

    def test_custom_threshold(self):
        y_pred = torch.tensor([0.3, 0.7])
        y_true = torch.tensor([1.0, 1.0])
        acc = binary_accuracy(y_pred, y_true, threshold=0.2)
        assert acc.item() == pytest.approx(1.0)


class TestPrecision:
    def test_perfect_precision(self):
        y_pred = torch.tensor([0.9, 0.1, 0.8])
        y_true = torch.tensor([1.0, 0.0, 1.0])
        p = precision(y_pred, y_true)
        assert p.item() == pytest.approx(1.0, abs=1e-5)

    def test_zero_precision(self):
        y_pred = torch.tensor([0.9, 0.9])
        y_true = torch.tensor([0.0, 0.0])
        p = precision(y_pred, y_true)
        assert p.item() == pytest.approx(0.0, abs=1e-5)

    def test_mixed_predictions(self):
        y_pred = torch.tensor([0.9, 0.9])
        y_true = torch.tensor([1.0, 0.0])
        p = precision(y_pred, y_true)
        assert p.item() == pytest.approx(0.5, abs=1e-4)

    def test_no_positive_predictions(self):
        y_pred = torch.tensor([0.1, 0.2])
        y_true = torch.tensor([1.0, 1.0])
        p = precision(y_pred, y_true)
        assert p.item() == pytest.approx(0.0, abs=1e-5)


class TestRecall:
    def test_perfect_recall(self):
        y_pred = torch.tensor([0.9, 0.1, 0.8])
        y_true = torch.tensor([1.0, 0.0, 1.0])
        r = recall(y_pred, y_true)
        assert r.item() == pytest.approx(1.0, abs=1e-5)

    def test_zero_recall(self):
        y_pred = torch.tensor([0.1, 0.1])
        y_true = torch.tensor([1.0, 1.0])
        r = recall(y_pred, y_true)
        assert r.item() == pytest.approx(0.0, abs=1e-5)

    def test_mixed_recall(self):
        y_pred = torch.tensor([0.9, 0.1])
        y_true = torch.tensor([1.0, 1.0])
        r = recall(y_pred, y_true)
        assert r.item() == pytest.approx(0.5, abs=1e-4)


class TestF1Score:
    def test_perfect_f1(self):
        y_pred = torch.tensor([0.9, 0.1, 0.8])
        y_true = torch.tensor([1.0, 0.0, 1.0])
        f1 = f1_score(y_pred, y_true)
        assert f1.item() == pytest.approx(1.0, abs=1e-5)

    def test_zero_f1(self):
        y_pred = torch.tensor([0.9, 0.9])
        y_true = torch.tensor([0.0, 0.0])
        f1 = f1_score(y_pred, y_true)
        assert f1.item() == pytest.approx(0.0, abs=1e-5)

    def test_f1_between_precision_and_recall(self):
        y_pred = torch.tensor([0.9, 0.9, 0.1])
        y_true = torch.tensor([1.0, 0.0, 1.0])
        f1 = f1_score(y_pred, y_true)
        p = precision(y_pred, y_true)
        r = recall(y_pred, y_true)
        expected = 2 * (p * r) / (p + r + 1e-7)
        assert f1.item() == pytest.approx(expected.item(), abs=1e-5)


class TestRegressionMetrics:
    def test_mse_perfect(self):
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.0, 2.0, 3.0])
        assert mse(y_pred, y_true).item() == pytest.approx(0.0)

    def test_mse_value(self):
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([2.0, 4.0])
        mse_val = mse(y_pred, y_true).item()
        assert mse_val == pytest.approx(10.0)

    def test_mae_perfect(self):
        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([1.0, 2.0])
        assert mae(y_pred, y_true).item() == pytest.approx(0.0)

    def test_mae_value(self):
        y_pred = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([2.0, 4.0])
        mae_val = mae(y_pred, y_true).item()
        assert mae_val == pytest.approx(3.0)

    def test_rmse_perfect(self):
        y_pred = torch.tensor([1.0, 2.0])
        y_true = torch.tensor([1.0, 2.0])
        assert rmse(y_pred, y_true).item() == pytest.approx(0.0)

    def test_rmse_value(self):
        y_pred = torch.tensor([0.0])
        y_true = torch.tensor([3.0])
        rmse_val = rmse(y_pred, y_true).item()
        assert rmse_val == pytest.approx(3.0)

    def test_rmse_always_non_negative(self):
        y_pred = torch.randn(20)
        y_true = torch.randn(20)
        assert rmse(y_pred, y_true).item() >= 0


class TestAccuracy:
    def test_multiclass_perfect(self):
        # logits where argmax matches target
        y_pred = torch.tensor([[10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, 10.0]])
        y_true = torch.tensor([0, 1, 2])
        assert accuracy(y_pred, y_true).item() == pytest.approx(1.0)

    def test_multiclass_all_wrong(self):
        y_pred = torch.tensor([[1.0, 10.0, 1.0], [10.0, 1.0, 1.0], [10.0, 1.0, 1.0]])
        y_true = torch.tensor([0, 1, 2])
        assert accuracy(y_pred, y_true).item() == pytest.approx(0.0)

    def test_multiclass_half(self):
        y_pred = torch.tensor([[10.0, 1.0], [1.0, 10.0], [10.0, 1.0], [1.0, 10.0]])
        y_true = torch.tensor([0, 1, 1, 0])
        assert accuracy(y_pred, y_true).item() == pytest.approx(0.5)

    def test_binary_fallback(self):
        y_pred = torch.tensor([0.9, 0.1, 0.8])
        y_true = torch.tensor([1.0, 0.0, 1.0])
        assert accuracy(y_pred, y_true).item() == pytest.approx(1.0)


class TestTopKAccuracy:
    def test_top1_perfect(self):
        y_pred = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
        y_true = torch.tensor([0])
        assert top_k_accuracy(y_pred, y_true, k=1).item() == pytest.approx(1.0)

    def test_top1_miss(self):
        y_pred = torch.tensor([[1.0, 10.0, 1.0, 1.0]])
        y_true = torch.tensor([0])
        assert top_k_accuracy(y_pred, y_true, k=1).item() == pytest.approx(0.0)

    def test_top3_hit(self):
        # target=3 is 4th highest, but within top-3? No.
        # Actually let's make it clear:
        y_pred = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])  # order: 1,3,2,4,0
        y_true = torch.tensor([3])  # index 3, value 4.0 → rank 2
        assert top_k_accuracy(y_pred, y_true, k=3).item() == pytest.approx(1.0)

    def test_top5_convenience(self):
        y_pred = torch.randn(8, 100)
        y_true = torch.randint(0, 100, (8,))
        val = top_5_accuracy(y_pred, y_true)
        assert 0 <= val.item() <= 1

    def test_top10_convenience(self):
        y_pred = torch.randn(8, 100)
        y_true = torch.randint(0, 100, (8,))
        val = top_10_accuracy(y_pred, y_true)
        assert 0 <= val.item() <= 1

    def test_top20_convenience(self):
        y_pred = torch.randn(8, 100)
        y_true = torch.randint(0, 100, (8,))
        val = top_20_accuracy(y_pred, y_true)
        assert 0 <= val.item() <= 1

    def test_topk_monotonic(self):
        """top_k should be non-decreasing as k increases."""
        y_pred = torch.randn(32, 50)
        y_true = torch.randint(0, 50, (32,))
        t5 = top_k_accuracy(y_pred, y_true, k=5).item()
        t10 = top_k_accuracy(y_pred, y_true, k=10).item()
        t20 = top_k_accuracy(y_pred, y_true, k=20).item()
        assert t5 <= t10 + 1e-6
        assert t10 <= t20 + 1e-6


class TestNDCG:
    def test_perfect_rank(self):
        """Target at rank 1 → NDCG = 1/log2(2) = 1.0."""
        y_pred = torch.tensor([[10.0, 1.0, 1.0]])
        y_true = torch.tensor([0])
        assert ndcg_at_k(y_pred, y_true, k=5).item() == pytest.approx(1.0)

    def test_rank2(self):
        """Target at rank 2 → NDCG = 1/log2(3)."""
        y_pred = torch.tensor([[5.0, 10.0, 1.0]])
        y_true = torch.tensor([0])  # index 0 has score 5.0 → rank 2
        expected = 1.0 / math.log2(3)
        assert ndcg_at_k(y_pred, y_true, k=5).item() == pytest.approx(expected, abs=1e-5)

    def test_miss(self):
        """Target not in top-k → NDCG = 0."""
        y_pred = torch.tensor([[1.0, 10.0, 9.0, 8.0, 7.0]])
        y_true = torch.tensor([0])  # index 0 has lowest score → rank 5
        assert ndcg_at_k(y_pred, y_true, k=2).item() == pytest.approx(0.0)

    def test_ndcg_monotonic(self):
        y_pred = torch.randn(32, 50)
        y_true = torch.randint(0, 50, (32,))
        n5 = ndcg_at_5(y_pred, y_true).item()
        n10 = ndcg_at_10(y_pred, y_true).item()
        n20 = ndcg_at_20(y_pred, y_true).item()
        assert n5 <= n10 + 1e-6
        assert n10 <= n20 + 1e-6

    def test_1d_input_raises_value_error(self):
        """1D ``y_pred`` is ambiguous for ranking metrics. The implementation
        deliberately raises rather than silently returning 0 — the previous
        silent fallback was indistinguishable from ``accuracy`` and masked
        a real schema bug. Callers should route binary models through
        binary_accuracy/precision/recall/f1/roc_auc instead.
        """
        y_pred = torch.tensor([0.5, 0.8])
        y_true = torch.tensor([1.0, 0.0])
        with pytest.raises(ValueError, match="multi-class logits"):
            ndcg_at_10(y_pred, y_true)


class TestMRR:
    def test_perfect_rank(self):
        """Target at rank 1 → MRR = 1.0."""
        y_pred = torch.tensor([[10.0, 1.0, 1.0]])
        y_true = torch.tensor([0])
        assert mrr_at_k(y_pred, y_true, k=5).item() == pytest.approx(1.0)

    def test_rank3(self):
        """Target at rank 3 → MRR = 1/3."""
        y_pred = torch.tensor([[5.0, 10.0, 8.0, 1.0]])
        y_true = torch.tensor([0])  # index 0 has score 5.0 → rank 3 (after 1,2)
        assert mrr_at_k(y_pred, y_true, k=5).item() == pytest.approx(1.0 / 3, abs=1e-5)

    def test_miss(self):
        """Target not in top-k → MRR = 0."""
        y_pred = torch.tensor([[1.0, 10.0, 9.0]])
        y_true = torch.tensor([0])
        assert mrr_at_k(y_pred, y_true, k=1).item() == pytest.approx(0.0)

    def test_mrr10_convenience(self):
        y_pred = torch.randn(8, 100)
        y_true = torch.randint(0, 100, (8,))
        val = mrr_at_10(y_pred, y_true)
        assert 0 <= val.item() <= 1


# ---------------------------------------------------------------------------
# ROC-AUC (post-training, sklearn-based) — tested at the function level
# ---------------------------------------------------------------------------
class TestComputeRocAuc:
    """Tests for the _compute_roc_auc logic extracted as pure functions.

    These mirror the logic in SupervisedModelTraining._compute_roc_auc()
    without requiring a full training pipeline.
    """

    def test_multiclass_perfect(self):
        """Perfect logits → AUC ≈ 1.0."""
        from sklearn.metrics import roc_auc_score
        import numpy as np

        n, c = 100, 5
        y_true = np.random.randint(0, c, size=n)
        # Perfect logits: high score for correct class
        logits = np.full((n, c), -10.0)
        logits[np.arange(n), y_true] = 10.0
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        y_onehot = np.zeros((n, c), dtype=int)
        y_onehot[np.arange(n), y_true] = 1

        auc = roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro")
        assert auc == pytest.approx(1.0, abs=1e-6)

    def test_multiclass_random(self):
        """Random logits → AUC ≈ 0.5."""
        from sklearn.metrics import roc_auc_score
        import numpy as np

        np.random.seed(42)
        n, c = 5000, 10
        y_true = np.random.randint(0, c, size=n)
        logits = np.random.randn(n, c)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        y_onehot = np.zeros((n, c), dtype=int)
        y_onehot[np.arange(n), y_true] = 1

        auc = roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro")
        assert 0.4 < auc < 0.6

    def test_binary_roc_auc(self):
        """Binary case: sigmoid scores → valid AUC."""
        from sklearn.metrics import roc_auc_score
        import numpy as np

        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_scores = np.array([0.1, 0.3, 0.9, 0.8, 0.7, 0.2, 0.6, 0.4])
        auc = roc_auc_score(y_true, y_scores)
        assert 0.0 <= auc <= 1.0
        assert auc > 0.8  # scores are well-ordered

    def test_present_classes_filter(self):
        """Only classes present in y_true should be used (OvR still works)."""
        from sklearn.metrics import roc_auc_score
        import numpy as np

        n, c = 60, 5
        # Only use classes 0, 1, 2 (not 3 or 4)
        y_true = np.random.choice([0, 1, 2], size=n)
        logits = np.random.randn(n, c)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        present = np.unique(y_true)
        y_onehot = np.zeros((n, c), dtype=int)
        y_onehot[np.arange(n), y_true] = 1

        auc = roc_auc_score(
            y_onehot[:, present],
            probs[:, present],
            multi_class="ovr",
            average="macro",
        )
        assert 0.0 <= auc <= 1.0

    def test_fewer_than_2_classes_skipped(self):
        """If only 1 class present, ROC-AUC is undefined → should skip."""
        import numpy as np

        y_true = np.array([3, 3, 3, 3])
        present = np.unique(y_true)
        assert len(present) < 2  # Should not attempt AUC
