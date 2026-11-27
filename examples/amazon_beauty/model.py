"""
Two-tower content-based ranker for Amazon Beauty.

User tower:  user_id  → Embedding → MLP → R^d
Item tower:  item_text_emb (precomputed) → MLP → R^d
Score:       <u, v>

The item tower has NO per-item-id parameter — it relies entirely on the
text embedding. That choice is deliberate: it lets the model generalise
to cold items (any new product with a text description) while still
modelling stable user preferences.
"""

from typing import Any, Dict

import torch.nn as nn

from mars_rec.meta_config import ProjectConfig
from mars_rec.model.abstract import RecommenderModule


class AmazonBeautyTwoTower(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int = 64,
        text_emb_dim: int = 64,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__(project_config, index_mapping)

        self.user_emb = nn.Embedding(self._n_users, n_factors)
        self.user_tower = nn.Sequential(
            nn.Linear(n_factors, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_factors),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(text_emb_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_factors),
        )

    def forward(self, *args):
        # args order from ProjectConfig.input_columns:
        #   0: user_id        (B,)
        #   1: item_id        (B,)        — not used by the item tower
        #   2: item_text_emb  (B, D_text) — looked up by item index
        user_ids = args[0]
        item_text_emb = args[2]

        u = self.user_tower(self.user_emb(user_ids))
        v = self.item_tower(item_text_emb)
        return (u * v).sum(dim=-1)
