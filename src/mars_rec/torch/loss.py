"""Custom loss functions for supervised recsys training.

Registered in
:data:`mars_rec.simulation.training.TORCH_LOSS_FUNCTIONS` so they
can be selected via the ``loss_function`` Luigi parameter
(``focal_loss``, ``implicit_feedback_bce``, ``bpr``, ``ranknet``,
``lambdarank``, ``bce_logits_smooth``, ``infonce``,
``cosine_similarity``, ``hardneg_infonce``, ``dummy``).
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class FocalLoss(nn.Module):
    """Focal loss — supports both binary (1D logits) and multi-class (2D probs).

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Inputs:
        - 1D `[B]`  → logits; sigmoid is applied internally (binary path,
          aligned with `lightning_module._LOGIT_LOSSES`).
        - 2D `[B, C]` → assumed already-softmaxed probabilities (legacy
          multi-class path; preserved for backwards compatibility).

    Reference: Lin et al., Focal Loss for Dense Object Detection,
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        epsilon = 1.0e-9
        t = targets.to(torch.float32)

        if inputs.dim() == 1:
            # Binary: inputs are logits.
            p = torch.sigmoid(inputs.to(torch.float32))
        else:
            # Multi-class legacy: inputs are probabilities.
            p = inputs.to(torch.float32) + epsilon

        pt = p * t + (1 - p) * (1 - t)
        # clamp avoids log(0) when sigmoid saturates (very negative logit + t=1)
        pt = pt.clamp(min=epsilon)
        ce = -torch.log(pt)
        weight = (1.0 - pt) ** self.gamma
        loss = weight * self.alpha * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ImplicitFeedbackBCELoss(nn.Module):
    """Confidence-weighted BCE for implicit-feedback datasets.

    Multiplies the per-row BCE by ``1 + sum_i w_i * c_i``, where the
    ``c_i`` come in as auxiliary target tensors (e.g. play count,
    dwell time) declared as :attr:`ProjectConfig.auxiliar_output_columns`
    and ``w_i`` are the corresponding entries of ``confidence_weights``.
    Inspired by Hu et al. 2008 "Collaborative Filtering for Implicit
    Feedback Datasets".
    """

    def __init__(self, confidence_weights: List[float], weight=None, reduction="mean") -> None:
        super().__init__()
        self.confidence_weights = confidence_weights
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, *confidence_targets: torch.Tensor) -> torch.Tensor:
        assert len(self.confidence_weights) == len(confidence_targets)
        confidence = torch.ones_like(target)
        for confidence_target, confidence_weight in zip(confidence_targets, self.confidence_weights):
            confidence += confidence_weight * confidence_target
        loss = confidence * F.binary_cross_entropy(input, target, weight=self.weight, reduction="none")
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DummyLoss(_Loss):
    """Passes the model's own loss tensor through unchanged (just reduces it).

    Useful when the model computes its own loss internally and returns
    it from ``forward`` — e.g. autoencoder-style training. Pair with
    ``loss_function="dummy"`` and have ``forward`` return the scalar
    loss directly (or a per-row vector when ``reduction != "none"``).
    """

    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)

    def forward(self, loss, target):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss


class CosineSimilarityLoss(nn.Module):
    """1 − cosine_similarity, averaged over the batch.

    Both ``pred_emb`` and ``target_emb`` are expected to be L2-normalised
    before being passed in (the dot-product then equals cosine similarity).
    """

    def forward(self, pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        cos_sim = (pred_emb * target_emb).sum(dim=-1)  # [B]
        return (1.0 - cos_sim).mean()


class InfoNCELoss(nn.Module):
    """In-batch contrastive loss (SimCLR / CLIP style).

    Each sample ``i`` treats ``target_emb[i]`` as its positive and all other
    ``target_emb[j≠i]`` as negatives.  Both tensors must be L2-normalised.

    Reference: Oord et al., "Representation Learning with Contrastive
    Predictive Coding", arXiv:1807.03748.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        # [B, B] similarity matrix; diagonal = positive pairs
        sim_matrix = pred_emb @ target_emb.T / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        return F.cross_entropy(sim_matrix, labels)


class HardNegInfoNCELoss(nn.Module):
    """InfoNCE with **per-query hard negatives** in addition to in-batch negs.

    Standard InfoNCE uses (B-1) in-batch negatives per query, which are
    semantically random for large vocabularies — easy negatives that the
    model can discriminate trivially once the embedding space is roughly
    aligned. Adding K hard negatives per query (e.g. CF-BPR ANN top-K
    neighbors of the positive) forces the model to learn fine-grained
    discrimination among items that are *already* close in the prior space.

    Forward signature::

        forward(pred_emb [B, D], target_emb [B, D],
                hard_neg_embs [B, K, D] | None) -> scalar

    All inputs must be L2-normalised. When ``hard_neg_embs`` is ``None``
    falls back to standard InfoNCE.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        pred_emb: torch.Tensor,
        target_emb: torch.Tensor,
        hard_neg_embs: torch.Tensor = None,
    ) -> torch.Tensor:
        in_batch_sim = pred_emb @ target_emb.T / self.temperature  # [B, B]
        labels = torch.arange(in_batch_sim.size(0), device=in_batch_sim.device)
        if hard_neg_embs is None:
            return F.cross_entropy(in_batch_sim, labels)
        # hard_neg_sim per query: [B, K]
        hard_sim = (pred_emb.unsqueeze(1) * hard_neg_embs).sum(-1) / self.temperature
        logits = torch.cat([in_batch_sim, hard_sim], dim=1)  # [B, B + K]
        return F.cross_entropy(logits, labels)


class BPRLoss(nn.Module):
    """Pairwise BPR — pointwise-batch adapter.

    Standard BPR is :math:`L = -\\log \\sigma(s_{pos} - s_{neg})`. The
    mars-rec pipeline emits flat ``(features, label)`` rows where
    ``label ∈ {0, 1}`` (positives interleaved with sampled negatives — DNS,
    emb-hard, popularity, etc.). To stay drop-in compatible with that
    pointwise format, this loss:

    1. Splits the batch by label.
    2. For each positive, draws one negative score uniformly from the same
       batch (with replacement).
    3. Computes ``-logsigmoid(s_pos - s_neg).mean()``.

    Returns ``0`` (preserving the autograd graph) if a batch happens to
    contain no positives or no negatives.

    Inputs are interpreted as raw logits (no sigmoid applied) — pair with
    the same models used for ``bce_logits`` / ``bce_logits_smooth``. Add
    ``"bpr"`` to ``_LOGIT_LOSSES`` so metrics that expect probabilities
    receive ``sigmoid(score)``.

    Reference: Rendle et al., *BPR: Bayesian Personalized Ranking from
    Implicit Feedback* (UAI 2009).
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}")
        self.reduction = reduction
        # Fallback for single-label batches (mostly validation: the val
        # DataLoader uses shuffle=False, and DNS-style datasets lay out
        # positives [0, n_pos) then negatives [n_pos, total), so most val
        # batches are pure positives or pure negatives → BPR has no pair to
        # form. BCE-with-logits is the standard pointwise proxy and stays
        # monotonic with ranking quality, so val_loss remains an honest
        # early-stopping signal.
        self._bce_fallback = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = scores.float()
        target = target.float()
        pos_mask = target > 0.5
        n_pos = int(pos_mask.sum().item())
        n_neg = int((~pos_mask).sum().item())
        if n_pos == 0 or n_neg == 0:
            # Single-label batch — use BCE so val_loss stays meaningful.
            return self._bce_fallback(scores, target)
        pos_scores = scores[pos_mask]  # [n_pos]
        neg_scores = scores[~pos_mask]  # [n_neg]
        idx = torch.randint(0, n_neg, (n_pos,), device=scores.device)
        sampled_neg = neg_scores[idx]  # [n_pos]
        loss_per = -F.logsigmoid(pos_scores - sampled_neg)
        if self.reduction == "mean":
            return loss_per.mean()
        if self.reduction == "sum":
            return loss_per.sum()
        return loss_per


class RankNetLoss(nn.Module):
    """In-batch all-pairs RankNet (Burges et al. 2005).

    For each batch, splits rows by ``target`` into positives (``label > 0.5``)
    and negatives, then forms the full cartesian product of pairs and applies
    the RankNet pairwise loss::

        L = mean_{i ∈ pos, j ∈ neg} −log σ(σ_param · (s_i − s_j))

    With ``sigma=1.0`` and a single (pos, neg) pair, this is numerically
    identical to ``BPRLoss``; the difference is that ``RankNetLoss`` uses
    **all** ``n_pos × n_neg`` pairs available in the batch, giving a denser
    gradient signal — closer to the original RankNet formulation. Useful when
    DNS-style mining provides multiple in-batch negatives per positive but
    BPR's single-pair sampling under-utilises the signal.

    Inputs are interpreted as raw logits (no sigmoid applied) — pair with the
    same models used for ``bce_logits`` / ``bpr``. ``ranknet`` is added to the
    framework's ``_LOGIT_LOSSES`` so metrics that expect probabilities receive
    ``sigmoid(score)``.

    Single-label batches (no positives or no negatives) fall back to
    ``BCEWithLogitsLoss`` so ``val_loss`` stays a meaningful early-stopping
    signal — same rationale as ``BPRLoss``.

    Reference: Burges et al., *Learning to Rank using Gradient Descent*
    (ICML 2005).
    """

    def __init__(self, sigma: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction={reduction!r}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = float(sigma)
        self.reduction = reduction
        self._bce_fallback = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = scores.float()
        target = target.float()
        pos_mask = target > 0.5
        n_pos = int(pos_mask.sum().item())
        n_neg = int((~pos_mask).sum().item())
        if n_pos == 0 or n_neg == 0:
            return self._bce_fallback(scores, target)
        s_pos = scores[pos_mask]  # [n_pos]
        s_neg = scores[~pos_mask]  # [n_neg]
        diff = s_pos.unsqueeze(1) - s_neg.unsqueeze(0)  # [n_pos, n_neg]
        loss_per_pair = -F.logsigmoid(self.sigma * diff)
        if self.reduction == "mean":
            return loss_per_pair.mean()
        return loss_per_pair.sum()


class LambdaRankLoss(nn.Module):
    """In-batch all-pairs LambdaRank with binary relevance (Burges 2010).

    Same all-pairs structure as :class:`RankNetLoss` but each pair is weighted
    by ``|ΔNDCG_ij|`` computed over the batch treated as a single pseudo-list:

        rank_k       = 1 + count(s_l > s_k for l in batch)      # 1-indexed
        DCG_k        = 1 / log2(rank_k + 1)
        IDCG         = sum_{k=1..n_pos} 1 / log2(k + 1)
        ΔNDCG_ij     = |DCG_i − DCG_j| / IDCG
        L            = mean_{i ∈ pos, j ∈ neg} ΔNDCG_ij · −log σ(σ · (s_i − s_j))

    Treating the whole batch as one list is the canonical mini-batch
    approximation of LambdaRank for multi-query mini-batches. The weighting
    causes pairs near the top of the ranking (small ``rank_i``) to contribute
    more, aligning the gradient with NDCG@k.

    The ΔNDCG weights are computed under ``torch.no_grad`` — gradient flows
    only through ``−log σ`` (Burges 2010 §6). This matches the LambdaRank
    paper recipe and avoids non-differentiable rank computation.

    Single-label batches (no positives or no negatives) fall back to
    ``BCEWithLogitsLoss``.

    Reference: Burges, *From RankNet to LambdaRank to LambdaMART: An Overview*
    (MSR-TR-2010-82).
    """

    def __init__(self, sigma: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction={reduction!r}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = float(sigma)
        self.reduction = reduction
        self._bce_fallback = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = scores.float()
        target = target.float()
        pos_mask = target > 0.5
        n_pos = int(pos_mask.sum().item())
        n_neg = int((~pos_mask).sum().item())
        if n_pos == 0 or n_neg == 0:
            return self._bce_fallback(scores, target)
        s_pos = scores[pos_mask]  # [n_pos]
        s_neg = scores[~pos_mask]  # [n_neg]

        with torch.no_grad():
            B = scores.numel()
            sorted_idx = scores.argsort(descending=True)
            ranks = torch.empty_like(sorted_idx)
            ranks[sorted_idx] = torch.arange(1, B + 1, device=scores.device)
            ranks_f = ranks.float()
            dcg_pos = 1.0 / torch.log2(ranks_f[pos_mask] + 1.0)  # [n_pos]
            dcg_neg = 1.0 / torch.log2(ranks_f[~pos_mask] + 1.0)  # [n_neg]
            ideal = torch.arange(
                1,
                n_pos + 1,
                dtype=torch.float,
                device=scores.device,
            )
            idcg = (1.0 / torch.log2(ideal + 1.0)).sum().clamp(min=1e-9)
            delta_ndcg = (dcg_pos.unsqueeze(1) - dcg_neg.unsqueeze(0)).abs() / idcg

        diff = s_pos.unsqueeze(1) - s_neg.unsqueeze(0)  # [n_pos, n_neg]
        loss_per_pair = -delta_ndcg * F.logsigmoid(self.sigma * diff)
        if self.reduction == "mean":
            return loss_per_pair.mean()
        return loss_per_pair.sum()


class BCEWithLogitsLossSmooth(nn.Module):
    """``BCEWithLogitsLoss`` with symmetric label smoothing.

    Smoothed targets: ``y' = y * (1 - α) + 0.5 * α`` so positives become
    ``1 - α/2`` and negatives become ``α/2``. This counters the
    "overconfidence trap" — models trained with vanilla BCE tend to drive
    sigmoid outputs to the saturated extremes (0 / 1), which hurts
    calibration on validation. Label smoothing keeps the optimum away from
    saturation, leading to flatter minima that generalise better.

    Reference: Müller et al., *When Does Label Smoothing Help?* (NeurIPS 2019).
    """

    def __init__(
        self,
        label_smoothing: float = 0.1,
        weight=None,
        pos_weight=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.label_smoothing = label_smoothing
        self._bce = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        a = self.label_smoothing
        # symmetric smoothing: 1 → 1 - a/2 ; 0 → a/2
        smoothed = target.float() * (1.0 - a) + 0.5 * a
        return self._bce(input, smoothed)
