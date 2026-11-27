"""Per-batch evaluation metrics consumed by Lightning during training.

Every callable registered in :data:`METRIC_REGISTRY` takes
``(y_pred, y_true)`` tensors and returns a scalar tensor. The
training task picks them by string name via the ``metrics`` Luigi
parameter (e.g. ``["loss", "binary_accuracy", "f1_score"]``).
"""

from typing import Sequence

import torch
import torch.nn.functional as F


def _preprocess(y_pred: torch.Tensor, y_true: torch.Tensor):
    """Common preprocessing: handle Sequence outputs and sparse tensors."""
    if isinstance(y_true, Sequence) and isinstance(y_pred, torch.Tensor):
        y_true = y_true[0]
    if y_true.layout == torch.sparse_coo:
        y_true = y_true.to_dense()
    return y_pred, y_true


def bce(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy. Forces FP32 under autocast for stability."""
    y_pred, y_true = _preprocess(y_pred, y_true)
    # ``binary_cross_entropy`` is autocast-unsafe (FP16 sigmoid + log can
    # underflow / produce inf); PyTorch refuses to dispatch it under an
    # active autocast context. Drop out of autocast and cast inputs to
    # FP32 so this metric works under ``precision="16-mixed"``.
    with torch.autocast(device_type=y_pred.device.type, enabled=False):
        loss = F.binary_cross_entropy(
            y_pred.float(),
            y_true.float(),
            reduction="none",
        )
    return loss.mean()


def binary_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Fraction of rows where ``y_pred > threshold`` agrees with ``y_true > threshold``."""
    y_pred, y_true = _preprocess(y_pred, y_true)
    y_pred = (y_pred.float() > threshold).long()
    y_true = (y_true.float() > threshold).long()
    return torch.eq(y_pred, y_true).view(-1).float().mean()


def precision(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Binary precision ``tp / (tp + fp)`` thresholded at ``threshold``."""
    y_pred, y_true = _preprocess(y_pred, y_true)
    y_pred = (y_pred.float() > threshold).float()
    y_true = (y_true.float() > threshold).float()

    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)

    epsilon = 1e-7
    return tp / (tp + fp + epsilon)


def recall(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Binary recall ``tp / (tp + fn)`` thresholded at ``threshold``."""
    y_pred, y_true = _preprocess(y_pred, y_true)
    y_pred = (y_pred.float() > threshold).float()
    y_true = (y_true.float() > threshold).float()

    tp = (y_true * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    return tp / (tp + fn + epsilon)


def f1_score(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Binary F1 score (harmonic mean of :func:`precision` and :func:`recall`)."""
    y_pred, y_true = _preprocess(y_pred, y_true)
    y_pred = (y_pred.float() > threshold).float()
    y_true = (y_true.float() > threshold).float()

    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    prec = tp / (tp + fp + epsilon)
    rec = tp / (tp + fn + epsilon)

    return 2 * (prec * rec) / (prec + rec + epsilon)


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean squared error (FP32-cast inputs)."""
    y_pred, y_true = _preprocess(y_pred, y_true)
    loss = F.mse_loss(y_pred.float(), y_true.float(), reduction="none")
    return loss.mean()


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    y_pred, y_true = _preprocess(y_pred, y_true)
    loss = torch.abs(y_pred.float() - y_true.float())
    return loss.mean()


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Root mean squared error."""
    y_pred, y_true = _preprocess(y_pred, y_true)
    loss = F.mse_loss(y_pred.float(), y_true.float(), reduction="mean")
    return torch.sqrt(loss)


# ---------------------------------------------------------------------------
# Multi-class / ranking metrics (work with logits [B, C] + targets [B])
# ---------------------------------------------------------------------------


def _is_multiclass(y_pred: torch.Tensor) -> bool:
    """Check if predictions are multi-class logits (2D with C > 1)."""
    return y_pred.dim() == 2 and y_pred.size(1) > 1


def top_k_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Top-k accuracy for multi-class logits [B, C] and targets [B].

    Returns the fraction of samples where the ground-truth class
    is among the top-k predicted classes. Raises ``ValueError`` on
    1D inputs: top-k ranking is undefined for binary heads, and the
    previous silent fallback to round-and-compare made this metric
    indistinguishable from ``accuracy`` on BCE models.
    """
    y_pred, y_true = _preprocess(y_pred, y_true)
    if not _is_multiclass(y_pred):
        raise ValueError(
            f"top_{k}_accuracy expects multi-class logits of shape [B, C] with C>1, "
            f"got y_pred.shape={tuple(y_pred.shape)}. For binary classification "
            f"models (single-logit head) use `binary_accuracy`, `precision`, `recall`, "
            f"`f1_score`, or `roc_auc` instead."
        )
    _, topk_indices = y_pred.topk(min(k, y_pred.size(1)), dim=1)
    targets = y_true.long().unsqueeze(1)
    hits = (topk_indices == targets).any(dim=1).float()
    return hits.mean()


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Exact-match accuracy: argmax(logits) == target for multi-class,
    or threshold-based for binary predictions.

    Note: for binary classification with class imbalance (e.g. 80% negatives
    via ``negative_proportion``) the trivial "always predict 0" baseline
    already scores ~80% — prefer ``precision``, ``recall``, ``f1_score``,
    or ``roc_auc`` for a meaningful signal.
    """
    y_pred, y_true = _preprocess(y_pred, y_true)
    if _is_multiclass(y_pred):
        predicted = y_pred.argmax(dim=1)
        return torch.eq(predicted, y_true.long()).float().mean()
    return torch.eq(y_pred.round().long(), y_true.long()).float().mean()


def top_5_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Top-5 accuracy (convenience wrapper)."""
    return top_k_accuracy(y_pred, y_true, k=5)


def top_10_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Top-10 accuracy (convenience wrapper)."""
    return top_k_accuracy(y_pred, y_true, k=10)


def top_20_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Top-20 accuracy (convenience wrapper)."""
    return top_k_accuracy(y_pred, y_true, k=20)


def ndcg_at_k(y_pred: torch.Tensor, y_true: torch.Tensor, k: int = 10) -> torch.Tensor:
    """NDCG@k for multi-class logits [B, C] and single-relevant-item targets [B].

    For each sample, checks where the ground-truth item ranks in the top-k
    predictions. NDCG = 1/log2(rank+1) if found, 0 otherwise.
    """
    y_pred, y_true = _preprocess(y_pred, y_true)
    if not _is_multiclass(y_pred):
        raise ValueError(
            f"ndcg_at_{k} expects multi-class logits of shape [B, C] with C>1, "
            f"got y_pred.shape={tuple(y_pred.shape)}. For binary classification, "
            f"use the offline ranking eval over the full item vocabulary."
        )
    k = min(k, y_pred.size(1))
    _, topk_indices = y_pred.topk(k, dim=1)
    targets = y_true.long().unsqueeze(1).expand_as(topk_indices)
    hits = topk_indices == targets  # [B, k]
    # positions are 1-indexed: first position = 1
    positions = torch.arange(1, k + 1, device=y_pred.device, dtype=torch.float32)
    dcg = (hits.float() / torch.log2(positions + 1)).sum(dim=1)  # [B]
    return dcg.mean()


def ndcg_at_5(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """NDCG@5 (convenience wrapper)."""
    return ndcg_at_k(y_pred, y_true, k=5)


def ndcg_at_10(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """NDCG@10 (convenience wrapper)."""
    return ndcg_at_k(y_pred, y_true, k=10)


def ndcg_at_20(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """NDCG@20 (convenience wrapper)."""
    return ndcg_at_k(y_pred, y_true, k=20)


def mrr_at_k(y_pred: torch.Tensor, y_true: torch.Tensor, k: int = 10) -> torch.Tensor:
    """MRR@k for multi-class logits [B, C] and single-relevant-item targets [B].

    Mean Reciprocal Rank: 1/rank of the first (only) relevant item in top-k.
    """
    y_pred, y_true = _preprocess(y_pred, y_true)
    if not _is_multiclass(y_pred):
        raise ValueError(
            f"mrr_at_{k} expects multi-class logits of shape [B, C] with C>1, "
            f"got y_pred.shape={tuple(y_pred.shape)}. For binary classification, "
            f"use the offline ranking eval over the full item vocabulary."
        )
    k = min(k, y_pred.size(1))
    _, topk_indices = y_pred.topk(k, dim=1)
    targets = y_true.long().unsqueeze(1).expand_as(topk_indices)
    hits = topk_indices == targets  # [B, k]
    positions = torch.arange(1, k + 1, device=y_pred.device, dtype=torch.float32)
    rr = (hits.float() / positions).sum(dim=1)  # [B] — at most one hit
    return rr.mean()


def mrr_at_10(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """MRR@10 (convenience wrapper)."""
    return mrr_at_k(y_pred, y_true, k=10)


def avg_cosine_similarity(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity between L2-normalised pred and target embeddings.

    Expects ``y_pred`` to be ``[B, D]`` (model output) and ``y_true`` to be
    ``[B]`` item indices (output_column=track_id, INDEXABLE).  The true
    embedding is looked up from the model via the LightningModule; here we
    receive the raw integer targets and skip the embedding lookup — the metric
    is logged directly in ``EmbeddingRecommenderLightningModule._shared_step``
    so this registry entry is a no-op placeholder that satisfies the framework.

    In practice, ``EmbeddingRecommenderLightningModule`` logs ``avg_cos_sim``
    directly and does not route through ``METRIC_REGISTRY``.  This entry is
    kept so the name can be referenced via ``--metrics avg_cosine_similarity``.
    """
    # If called with pre-computed similarity values passed as y_pred [B] scalar:
    if y_pred.dim() == 1:
        return y_pred.float().mean()
    # Fallback: dot-product of rows (works when y_true is also [B, D])
    if y_true.dim() == y_pred.dim() == 2:
        return F.cosine_similarity(y_pred.float(), y_true.float(), dim=-1).mean()
    return torch.tensor(0.0, device=y_pred.device)


METRIC_REGISTRY = {
    # Binary / regression
    "bce": bce,
    "binary_accuracy": binary_accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
    # Multi-class / ranking
    "accuracy": accuracy,
    "top_5_accuracy": top_5_accuracy,
    "top_10_accuracy": top_10_accuracy,
    "top_20_accuracy": top_20_accuracy,
    "ndcg@5": ndcg_at_5,
    "ndcg@10": ndcg_at_10,
    "ndcg@20": ndcg_at_20,
    "mrr@10": mrr_at_10,
    # Embedding similarity (used by EmbeddingRecommenderLightningModule)
    "avg_cosine_similarity": avg_cosine_similarity,
}
