"""Tests for embedding-based losses in ``mars_rec.torch.loss``.

Complements ``test_loss.py`` (which covers BCE/Focal/BPR/RankNet/LambdaRank)
with the contrastive / cosine family used by
``EmbeddingRecommenderLightningModule``:

- ``CosineSimilarityLoss`` — 1 − cosine_similarity over the batch.
- ``InfoNCELoss`` — SimCLR/CLIP-style contrastive over in-batch negatives.
- ``HardNegInfoNCELoss`` — InfoNCE + per-query hard negatives.
- ``ImplicitFeedbackBCELoss`` — confidence-weighted BCE.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from mars_rec.torch.loss import (
    CosineSimilarityLoss,
    HardNegInfoNCELoss,
    ImplicitFeedbackBCELoss,
    InfoNCELoss,
)


def _normalised(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, dim=-1)


# ---------------------------------------------------------------------------
# CosineSimilarityLoss
# ---------------------------------------------------------------------------

class TestCosineSimilarityLoss:
    def test_zero_when_predictions_equal_targets(self):
        torch.manual_seed(0)
        x = _normalised(torch.randn(4, 8))
        loss = CosineSimilarityLoss()(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_two_when_predictions_are_antiparallel(self):
        """Anti-parallel L2-normalised vectors give cosine = -1, so the loss
        ``1 - cosine`` = 2. Sanity check that the formula is right."""
        x = _normalised(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        y = -x
        loss = CosineSimilarityLoss()(x, y)
        assert loss.item() == pytest.approx(2.0, abs=1e-6)

    def test_output_is_scalar_and_in_zero_two_range(self):
        torch.manual_seed(1)
        x = _normalised(torch.randn(8, 16))
        y = _normalised(torch.randn(8, 16))
        loss = CosineSimilarityLoss()(x, y)
        assert loss.ndim == 0
        assert 0.0 <= loss.item() <= 2.0


# ---------------------------------------------------------------------------
# InfoNCELoss
# ---------------------------------------------------------------------------

class TestInfoNCELoss:
    def test_loss_is_lower_when_pairs_are_aligned(self):
        """When pred_emb == target_emb (diagonal similarity = 1, off-diag ~0),
        InfoNCE is at its minimum. Random pairs give a much larger loss."""
        torch.manual_seed(42)
        emb = _normalised(torch.randn(8, 16))
        aligned_loss = InfoNCELoss(temperature=0.1)(emb, emb).item()
        misaligned_loss = InfoNCELoss(temperature=0.1)(
            emb, _normalised(torch.randn(8, 16))
        ).item()
        assert aligned_loss < misaligned_loss

    def test_lower_temperature_sharpens_loss(self):
        """Smaller temperature makes the softmax sharper. For non-trivial
        pairs the loss responds — value at temp=0.05 differs from temp=1.0."""
        torch.manual_seed(7)
        pred = _normalised(torch.randn(4, 8))
        target = _normalised(torch.randn(4, 8))
        sharp = InfoNCELoss(temperature=0.05)(pred, target).item()
        soft = InfoNCELoss(temperature=1.0)(pred, target).item()
        assert sharp != soft

    def test_returns_scalar(self):
        emb = _normalised(torch.randn(5, 12))
        loss = InfoNCELoss()(emb, emb)
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# HardNegInfoNCELoss
# ---------------------------------------------------------------------------

class TestHardNegInfoNCELoss:
    def test_falls_back_to_infonce_when_hard_neg_is_none(self):
        """Without hard negatives, this loss must equal InfoNCE at the same
        temperature. Lets the framework swap freely between the two."""
        torch.manual_seed(0)
        pred = _normalised(torch.randn(6, 8))
        target = _normalised(torch.randn(6, 8))
        T = 0.3
        hard = HardNegInfoNCELoss(temperature=T)(pred, target, None).item()
        infonce = InfoNCELoss(temperature=T)(pred, target).item()
        assert hard == pytest.approx(infonce, abs=1e-6)

    def test_hard_negatives_change_loss(self):
        """Adding hard negatives produces a *different* loss value than the
        InfoNCE fallback — the K extra columns in the logits change the
        softmax denominator. Sanity check that the K argument is actually
        wired through."""
        torch.manual_seed(11)
        pred = _normalised(torch.randn(4, 8))
        target = _normalised(torch.randn(4, 8))
        hard_negs = _normalised(torch.randn(4, 3, 8))  # K=3 hard negs per query

        loss_with = HardNegInfoNCELoss(temperature=0.5)(pred, target, hard_negs).item()
        loss_without = HardNegInfoNCELoss(temperature=0.5)(pred, target, None).item()
        assert loss_with != loss_without

    def test_returns_scalar_with_hard_negs(self):
        pred = _normalised(torch.randn(2, 4))
        target = _normalised(torch.randn(2, 4))
        hard_negs = _normalised(torch.randn(2, 5, 4))
        loss = HardNegInfoNCELoss()(pred, target, hard_negs)
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# ImplicitFeedbackBCELoss
# ---------------------------------------------------------------------------

class TestImplicitFeedbackBCELoss:
    def test_confidence_targets_boost_loss(self):
        """Passing a confidence_target of all ones with weight 5 should
        roughly 6x the BCE (1 + 5*1 = 6). Tests the per-row weighting."""
        loss_fn = ImplicitFeedbackBCELoss(confidence_weights=[5.0])
        pred = torch.tensor([0.3, 0.7])
        target = torch.tensor([1.0, 0.0])
        conf = torch.ones(2)  # 1 extra confidence per row
        loss = loss_fn(pred, target, conf).item()
        # Baseline BCE (no confidence) = F.binary_cross_entropy(pred, target).mean()
        base = F.binary_cross_entropy(pred, target).item()
        # confidence = 1 + 5*1 = 6 → loss ≈ 6 * base
        assert loss == pytest.approx(6.0 * base, rel=1e-5)

    def test_zero_confidence_weights_recovers_bce(self):
        """With confidence_weights=[] (empty), the loss must equal plain BCE."""
        loss_fn = ImplicitFeedbackBCELoss(confidence_weights=[])
        pred = torch.tensor([0.2, 0.8])
        target = torch.tensor([0.0, 1.0])
        loss = loss_fn(pred, target).item()
        bce = F.binary_cross_entropy(pred, target).item()
        assert loss == pytest.approx(bce, rel=1e-6)

    def test_mismatched_confidence_count_raises_assertion(self):
        """``confidence_weights`` must have the same length as the
        ``*confidence_targets`` positional args — guards against silently
        ignoring weights at runtime."""
        loss_fn = ImplicitFeedbackBCELoss(confidence_weights=[1.0, 2.0])
        pred = torch.tensor([0.3])
        target = torch.tensor([1.0])
        with pytest.raises(AssertionError):
            loss_fn(pred, target, torch.ones(1))  # only 1 conf target for 2 weights
