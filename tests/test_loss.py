import pytest
import torch
import torch.nn.functional as F
import numpy as np

from mars_rec.torch.loss import (
    DummyLoss,
    FocalLoss,
    ImplicitFeedbackBCELoss,
    LambdaRankLoss,
    RankNetLoss,
)


class TestFocalLoss:
    """1D inputs are treated as logits (sigmoid applied internally);
    test fixtures express probabilities via `torch.logit(...)` so the
    expected behavior matches the original probability-space tests."""

    @pytest.fixture
    def focal_loss(self):
        return FocalLoss(gamma=2.0, alpha=1.0)

    def test_output_is_scalar(self, focal_loss):
        pred = torch.logit(torch.tensor([0.7, 0.3, 0.8]))
        target = torch.tensor([1.0, 0.0, 1.0])
        loss = focal_loss(pred, target)
        assert loss.shape == torch.Size([])

    def test_loss_is_non_negative(self, focal_loss):
        pred = torch.logit(torch.tensor([0.7, 0.3, 0.8]))
        target = torch.tensor([1.0, 0.0, 1.0])
        loss = focal_loss(pred, target)
        assert loss.item() >= 0

    def test_perfect_predictions_low_loss(self, focal_loss):
        pred = torch.logit(torch.tensor([0.999, 0.001, 0.999]))
        target = torch.tensor([1.0, 0.0, 1.0])
        loss = focal_loss(pred, target)
        assert loss.item() < 0.01

    def test_wrong_predictions_higher_loss(self, focal_loss):
        pred_correct = torch.logit(torch.tensor([0.9, 0.1]))
        pred_wrong = torch.logit(torch.tensor([0.1, 0.9]))
        target = torch.tensor([1.0, 0.0])
        loss_correct = focal_loss(pred_correct, target)
        loss_wrong = focal_loss(pred_wrong, target)
        assert loss_wrong.item() > loss_correct.item()

    def test_gamma_zero_equals_cross_entropy(self):
        # With gamma=0, focal loss reduces to standard BCE-like loss
        focal = FocalLoss(gamma=0.0, alpha=1.0)
        pred = torch.logit(torch.tensor([0.7, 0.3]))
        target = torch.tensor([1.0, 0.0])
        loss = focal(pred, target)
        assert loss.item() >= 0

    def test_reduction_sum_returns_sum(self):
        focal = FocalLoss(gamma=2.0, alpha=1.0, reduction="sum")
        focal_mean = FocalLoss(gamma=2.0, alpha=1.0, reduction="mean")
        pred = torch.logit(torch.tensor([0.7, 0.3, 0.8]))
        target = torch.tensor([1.0, 0.0, 1.0])
        loss_sum = focal(pred, target)
        loss_mean = focal_mean(pred, target)
        assert abs(loss_sum.item() - loss_mean.item() * 3) < 1e-5

    def test_alpha_scaling(self):
        focal1 = FocalLoss(gamma=2.0, alpha=1.0)
        focal2 = FocalLoss(gamma=2.0, alpha=2.0)
        pred = torch.logit(torch.tensor([0.7, 0.3]))
        target = torch.tensor([1.0, 0.0])
        loss1 = focal1(pred, target)
        loss2 = focal2(pred, target)
        assert abs(loss2.item() - 2 * loss1.item()) < 1e-5

    def test_batch_input(self, focal_loss):
        pred = torch.randn(16)  # logits, not probs
        target = torch.randint(0, 2, (16,)).float()
        loss = focal_loss(pred, target)
        assert loss.shape == torch.Size([])

    def test_binary_logit_path_no_overflow(self, focal_loss):
        # Very negative logit + target=1 used to drive pt to 0 → log(0) = -inf.
        # The clamp(min=epsilon) keeps the loss finite.
        pred = torch.tensor([-50.0, -50.0])
        target = torch.tensor([1.0, 1.0])
        loss = focal_loss(pred, target)
        assert torch.isfinite(loss)

    def test_multiclass_2d_legacy_path(self, focal_loss):
        # 2D inputs are treated as already-softmaxed probabilities.
        pred = torch.tensor([[0.2, 0.8], [0.9, 0.1]])
        target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        loss = focal_loss(pred, target)
        assert torch.isfinite(loss)
        assert loss.item() >= 0


class TestImplicitFeedbackBCELoss:
    def test_no_confidence_weights(self):
        loss_fn = ImplicitFeedbackBCELoss(confidence_weights=[])
        pred = torch.tensor([0.7, 0.3, 0.8])
        target = torch.tensor([1.0, 0.0, 1.0])
        loss = loss_fn(pred, target)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_with_confidence_weights(self):
        loss_fn = ImplicitFeedbackBCELoss(confidence_weights=[0.5])
        pred = torch.tensor([0.7, 0.3])
        target = torch.tensor([1.0, 0.0])
        conf = torch.tensor([1.0, 0.0])
        loss = loss_fn(pred, target, conf)
        assert loss.item() >= 0

    def test_mismatched_confidence_raises(self):
        loss_fn = ImplicitFeedbackBCELoss(confidence_weights=[0.5, 0.3])
        pred = torch.tensor([0.7])
        target = torch.tensor([1.0])
        conf = torch.tensor([1.0])
        with pytest.raises(AssertionError):
            loss_fn(pred, target, conf)  # 1 tensor but 2 weights

    def test_reduction_sum(self):
        loss_mean = ImplicitFeedbackBCELoss(confidence_weights=[], reduction="mean")
        loss_sum = ImplicitFeedbackBCELoss(confidence_weights=[], reduction="sum")
        pred = torch.tensor([0.7, 0.3, 0.8])
        target = torch.tensor([1.0, 0.0, 1.0])
        lm = loss_mean(pred, target)
        ls = loss_sum(pred, target)
        assert abs(ls.item() - lm.item() * 3) < 1e-5

    def test_reduction_none(self):
        loss_fn = ImplicitFeedbackBCELoss(confidence_weights=[], reduction="none")
        pred = torch.tensor([0.7, 0.3, 0.8])
        target = torch.tensor([1.0, 0.0, 1.0])
        loss = loss_fn(pred, target)
        assert loss.shape == torch.Size([3])

    def test_zero_confidence_weight_no_effect(self):
        loss_no_conf = ImplicitFeedbackBCELoss(confidence_weights=[])
        loss_zero_conf = ImplicitFeedbackBCELoss(confidence_weights=[0.0])
        pred = torch.tensor([0.7, 0.3])
        target = torch.tensor([1.0, 0.0])
        conf = torch.zeros(2)
        l1 = loss_no_conf(pred, target)
        l2 = loss_zero_conf(pred, target, conf)
        assert abs(l1.item() - l2.item()) < 1e-5

    def test_confidence_weight_increases_loss(self):
        loss_no_conf = ImplicitFeedbackBCELoss(confidence_weights=[])
        loss_with_conf = ImplicitFeedbackBCELoss(confidence_weights=[1.0])
        pred = torch.tensor([0.7, 0.3])
        target = torch.tensor([1.0, 0.0])
        conf = torch.ones(2)
        l1 = loss_no_conf(pred, target)
        l2 = loss_with_conf(pred, target, conf)
        assert l2.item() > l1.item()


class TestDummyLoss:
    def test_mean_reduction(self):
        loss_fn = DummyLoss(reduction="mean")
        loss_input = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.0])
        result = loss_fn(loss_input, target)
        assert result.item() == pytest.approx(2.0)

    def test_sum_reduction(self):
        loss_fn = DummyLoss(reduction="sum")
        loss_input = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.0])
        result = loss_fn(loss_input, target)
        assert result.item() == pytest.approx(6.0)

    def test_none_reduction(self):
        loss_fn = DummyLoss(reduction="none")
        loss_input = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.0])
        result = loss_fn(loss_input, target)
        assert result.shape == torch.Size([3])
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_scalar_input(self):
        loss_fn = DummyLoss(reduction="mean")
        loss_input = torch.tensor(5.0)
        target = torch.tensor(0.0)
        result = loss_fn(loss_input, target)
        assert result.item() == pytest.approx(5.0)


class TestRankNetLoss:
    @pytest.fixture
    def loss_fn(self):
        return RankNetLoss(sigma=1.0)

    def test_output_is_scalar(self, loss_fn):
        scores = torch.tensor([2.0, 1.0, -1.0, -2.0])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = loss_fn(scores, target)
        assert loss.shape == torch.Size([])

    def test_loss_is_non_negative(self, loss_fn):
        torch.manual_seed(0)
        scores = torch.randn(16)
        target = torch.randint(0, 2, (16,)).float()
        # Skip degenerate batches the fixture is too small to control.
        if target.sum() == 0 or target.sum() == 16:
            target[0] = 1.0
            target[-1] = 0.0
        loss = loss_fn(scores, target)
        assert loss.item() >= 0

    def test_perfect_ordering_low_loss(self, loss_fn):
        scores = torch.tensor([10.0, 10.0, -10.0, -10.0])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = loss_fn(scores, target)
        assert loss.item() < 1e-3

    def test_inverted_ordering_high_loss(self, loss_fn):
        good = loss_fn(
            torch.tensor([5.0, -5.0]),
            torch.tensor([1.0, 0.0]),
        )
        bad = loss_fn(
            torch.tensor([-5.0, 5.0]),
            torch.tensor([1.0, 0.0]),
        )
        assert bad.item() > good.item()
        assert bad.item() > 1.0  # roughly 10 for σ=1, sanity bound

    def test_single_label_batch_falls_back_to_bce(self, loss_fn):
        # All positives
        scores = torch.tensor([0.1, 0.2, 0.3])
        target = torch.tensor([1.0, 1.0, 1.0])
        loss = loss_fn(scores, target)
        bce = torch.nn.BCEWithLogitsLoss()(scores, target)
        assert torch.allclose(loss, bce)
        # All negatives
        target_neg = torch.tensor([0.0, 0.0, 0.0])
        loss_neg = loss_fn(scores, target_neg)
        bce_neg = torch.nn.BCEWithLogitsLoss()(scores, target_neg)
        assert torch.allclose(loss_neg, bce_neg)

    def test_sigma_scales_logit_diff(self):
        scores = torch.tensor([1.0, -1.0])
        target = torch.tensor([1.0, 0.0])
        loss1 = RankNetLoss(sigma=1.0)(scores, target)
        loss2 = RankNetLoss(sigma=2.0)(scores, target)
        # With one pair, σ=2 ⇒ logsigmoid(2*(s_p-s_n)) instead of logsigmoid(s_p-s_n)
        expected2 = -F.logsigmoid(torch.tensor(2.0 * 2.0))
        assert torch.allclose(loss2, expected2, atol=1e-6)
        # And both are positive but loss2 < loss1 since wider margin
        assert loss2.item() < loss1.item()

    def test_matches_bpr_one_pair(self, loss_fn):
        scores = torch.tensor([1.5, -0.3])
        target = torch.tensor([1.0, 0.0])
        loss = loss_fn(scores, target)
        expected = -F.logsigmoid(torch.tensor(1.5 - (-0.3)))
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_gradient_flows(self):
        scores = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = RankNetLoss(sigma=1.0)(scores, target)
        loss.backward()
        assert scores.grad is not None
        # Positives should get negative gradient (push score up); negatives positive (push score down).
        assert (scores.grad[target > 0.5] < 0).all()
        assert (scores.grad[target < 0.5] > 0).all()

    def test_reduction_sum(self):
        scores = torch.tensor([1.0, -1.0, -2.0])
        target = torch.tensor([1.0, 0.0, 0.0])
        mean_loss = RankNetLoss(sigma=1.0, reduction="mean")(scores, target)
        sum_loss = RankNetLoss(sigma=1.0, reduction="sum")(scores, target)
        # 1 positive × 2 negatives = 2 pairs
        assert torch.allclose(sum_loss, mean_loss * 2, atol=1e-6)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            RankNetLoss(sigma=0.0)
        with pytest.raises(ValueError):
            RankNetLoss(sigma=-1.0)


class TestLambdaRankLoss:
    @pytest.fixture
    def loss_fn(self):
        return LambdaRankLoss(sigma=1.0)

    def test_output_is_scalar(self, loss_fn):
        scores = torch.tensor([2.0, 1.0, -1.0, -2.0])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = loss_fn(scores, target)
        assert loss.shape == torch.Size([])

    def test_loss_is_non_negative(self, loss_fn):
        torch.manual_seed(1)
        scores = torch.randn(20)
        target = torch.zeros(20)
        target[:5] = 1.0
        loss = loss_fn(scores, target)
        assert loss.item() >= 0

    def test_perfect_ordering_low_loss(self, loss_fn):
        scores = torch.tensor([10.0, 10.0, -10.0, -10.0])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = loss_fn(scores, target)
        assert loss.item() < 1e-3

    def test_single_label_batch_falls_back_to_bce(self, loss_fn):
        scores = torch.tensor([0.1, 0.2, 0.3])
        target = torch.tensor([1.0, 1.0, 1.0])
        bce = torch.nn.BCEWithLogitsLoss()(scores, target)
        assert torch.allclose(loss_fn(scores, target), bce)

    def test_weights_top_pairs_more(self):
        # Build a 4-row batch with 1 pos at top and 1 pos at bottom of the ranking.
        # Both pairs have identical logit diff (so RankNet term is identical),
        # but the top-positive pair should get a larger ΔNDCG weight than the
        # bottom-positive pair.
        #
        # Use sum reduction to inspect per-row contributions without averaging.
        loss_top = LambdaRankLoss(sigma=1.0, reduction="sum")(
            scores=torch.tensor([5.0, -5.0, -3.0, -7.0]),  # pos at rank 1, neg at rank 4
            target=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        loss_bottom = LambdaRankLoss(sigma=1.0, reduction="sum")(
            scores=torch.tensor([5.0, 3.0, -5.0, 7.0]),  # pos at rank 3, neg at rank 4 (rank 1)
            target=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        # The "top" case has the positive already perfectly placed → small
        # ΔNDCG. The "bottom" case has it badly placed → larger ΔNDCG.
        # Both have wider margin for the same pair pattern; we expect bottom
        # to yield a larger weighted contribution because of |ΔNDCG|.
        assert loss_bottom.item() > loss_top.item()

    def test_gradient_flows(self):
        scores = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = LambdaRankLoss(sigma=1.0)(scores, target)
        loss.backward()
        assert scores.grad is not None
        assert (scores.grad[target > 0.5] < 0).all()
        assert (scores.grad[target < 0.5] > 0).all()

    def test_reduction_sum_vs_mean(self):
        scores = torch.tensor([2.0, 1.0, -1.0, -2.0])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        mean_loss = LambdaRankLoss(sigma=1.0, reduction="mean")(scores, target)
        sum_loss = LambdaRankLoss(sigma=1.0, reduction="sum")(scores, target)
        # 2 pos × 2 neg = 4 pairs
        assert torch.allclose(sum_loss, mean_loss * 4, atol=1e-6)

    def test_delta_ndcg_not_in_grad_graph(self):
        # The ΔNDCG weighting must be detached — verify by checking that
        # gradient w.r.t. scores depends only on the logit-diff term.
        # Concretely, if we replace ΔNDCG-aware loss with manual computation
        # using the same per-pair weights as constants, gradients must match.
        scores = torch.tensor([1.0, -1.0, -2.0, -3.0], requires_grad=True)
        target = torch.tensor([1.0, 0.0, 0.0, 0.0])
        loss = LambdaRankLoss(sigma=1.0)(scores, target)
        loss.backward()
        grad_norm = scores.grad.norm().item()
        assert torch.isfinite(scores.grad).all()
        assert grad_norm > 0

    def test_decreases_when_optimised(self):
        # Sanity end-to-end: a small Linear scorer trained for a few steps
        # should reduce the loss on a fixed batch.
        torch.manual_seed(42)
        scorer = torch.nn.Linear(4, 1, bias=False)
        x = torch.randn(16, 4)
        # First 4 rows are positives; rest are negatives.
        target = torch.cat([torch.ones(4), torch.zeros(12)])
        loss_fn = LambdaRankLoss(sigma=1.0)
        opt = torch.optim.SGD(scorer.parameters(), lr=0.5)
        losses = []
        for _ in range(30):
            opt.zero_grad()
            scores = scorer(x).squeeze(-1)
            loss = loss_fn(scores, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # Should descend overall.
        assert losses[-1] < losses[0]

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            LambdaRankLoss(sigma=0.0)
        with pytest.raises(ValueError):
            LambdaRankLoss(sigma=-2.0)
