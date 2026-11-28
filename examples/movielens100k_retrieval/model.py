"""
Two-tower retrieval model for MovieLens 100K, trained with InfoNCE.

Query tower:  user_emb(user) ⊕ mean(item_emb(history))  → MLP → L2-normalise
Item tower:   item_emb(item)                             → MLP → L2-normalise
Score:        <query, item>  (= cosine similarity)

Embedding models expose two methods to the framework:

- ``forward(*inputs)`` → the query embedding ``[batch, dim]``;
- ``get_target_emb(item_ids)`` → the item embeddings ``[n, dim]``.

``EmbeddingRecommenderLightningModule`` feeds both to the loss, and the
ranking evaluation scores every catalog item with the same dot product.
Because the item tower depends only on the item id, all item vectors can be
precomputed and served from an approximate-nearest-neighbour index.

logQ correction (Yi et al., "Sampling-Bias-Corrected Neural Modeling for
Large Corpus Item Recommendations", RecSys 2019): with in-batch negatives a
popular item is a negative far more often than a rare one, so plain InfoNCE
learns to push popular items down, and loses to a most-popular baseline on a
full-catalog evaluation. The fix is to train on ``sim(q, j) / τ - log p(j)``,
where ``p(j)`` is the probability that item ``j`` appears in a batch. The
model estimates ``p`` by counting the target ids it embeds during training
and carries the correction in one extra embedding dimension:

    query = [q, 1]        item = [v_j, -τ · log p̂(j)]   (training)
                          item = [v_j, 0]               (evaluation)

so ``query · item / τ = q · v_j / τ - log p̂(j)`` inside the unchanged
``InfoNCELoss``, and the plain ``q · v_j`` is used for ranking. ``τ`` must
match the loss ``temperature``.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mars_rec.meta_config import ProjectConfig
from mars_rec.model.abstract import RecommenderModule

PAD_INDEX: int = 2  # "-1" maps to index slot 2 (see utils.index_mapping)


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))


class ML100KTwoTowerRetrieval(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int = 64,
        hidden_size: int = 128,
        dropout: float = 0.1,
        logq_correction: bool = True,
        temperature: float = 0.1,
    ):
        super().__init__(project_config, index_mapping)
        # One item table shared by the history (query side) and the targets.
        self.item_emb = nn.Embedding(self._n_items, n_factors, padding_idx=PAD_INDEX)
        self.user_emb = nn.Embedding(self._n_users, n_factors)
        self.query_tower = _mlp(2 * n_factors, hidden_size, n_factors, dropout)
        self.item_tower = _mlp(n_factors, hidden_size, n_factors, dropout)

        self.logq_correction = logq_correction
        self.temperature = temperature
        # Streaming count of each item as a training target (+1 smoothing).
        self.register_buffer("item_counts", torch.ones(self._n_items))

    def forward(self, user_ids: torch.Tensor, history_ids: torch.Tensor) -> torch.Tensor:
        # args follow ProjectConfig.input_columns (item_is_input=False):
        #   0: user_id          (B,)
        #   1: history_item_ids (B, L)
        mask = (history_ids != PAD_INDEX).unsqueeze(-1).float()  # (B, L, 1)
        history = (self.item_emb(history_ids) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        query = F.normalize(self.query_tower(torch.cat([self.user_emb(user_ids), history], dim=-1)), dim=-1)
        if not self.logq_correction:
            return query  # (B, d)
        return torch.cat([query, torch.ones_like(query[..., :1])], dim=-1)  # (B, d + 1)

    def get_target_emb(self, item_ids: torch.Tensor) -> torch.Tensor:
        item = F.normalize(self.item_tower(self.item_emb(item_ids)), dim=-1)  # (..., d)
        if not self.logq_correction:
            return item
        if self.training:
            # Only the training step embeds batch targets in train mode, so
            # these counts estimate how often each item is an in-batch negative.
            with torch.no_grad():
                self.item_counts.index_add_(
                    0, item_ids.reshape(-1), torch.ones(item_ids.numel(), device=item_ids.device)
                )
            log_p = torch.log(self.item_counts[item_ids] / self.item_counts.sum())
            correction = (-self.temperature * log_p).unsqueeze(-1)
        else:
            correction = torch.zeros_like(item[..., :1])
        return torch.cat([item, correction], dim=-1)  # (..., d + 1)
