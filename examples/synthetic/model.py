"""
Matrix factorisation with biases — the simplest useful recommender.

    score(u, i) = <p_u, q_i> + b_u + b_i + b

Returns one logit per row (``[batch]``), so it trains with ``bce_logits``.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from mars_rec.meta_config import ProjectConfig
from mars_rec.model.abstract import RecommenderModule


class SyntheticMF(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int = 16,
    ):
        super().__init__(project_config, index_mapping)
        self.user_emb = nn.Embedding(self._n_users, n_factors)
        self.item_emb = nn.Embedding(self._n_items, n_factors)
        self.user_bias = nn.Embedding(self._n_users, 1)
        self.item_bias = nn.Embedding(self._n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        for emb in (self.user_emb, self.item_emb):
            nn.init.normal_(emb.weight, std=0.1)
        for emb in (self.user_bias, self.item_bias):
            nn.init.zeros_(emb.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # args follow ProjectConfig.input_columns: user_id (B,), item_id (B,)
        dot = (self.user_emb(user_ids) * self.item_emb(item_ids)).sum(dim=-1)
        bias = self.user_bias(user_ids).squeeze(-1) + self.item_bias(item_ids).squeeze(-1)
        return dot + bias + self.global_bias  # [B] logits
