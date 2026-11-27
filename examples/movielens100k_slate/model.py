"""
Slate-ranking two-tower model for MovieLens 100K.

User tower:  user_id      → Embedding → MLP
Item tower:  item_id_emb  ⊕  item_genre_vec → MLP
Score:       <u, v>

Notes:
- The model treats negatives the same way as positives at the forward
  pass; the per-row slate logic lives in the dataset_class
  (``InteractionsWithNegativeItemGenerationByAvailableItemsDataset``).
- Item embedding combines a learnt id-emb with the content (genre) vector,
  letting cold-ish items rely on genre while well-seen items use the id.
"""
from typing import Any, Dict

import torch
import torch.nn as nn

from mars_rec.meta_config import ProjectConfig
from mars_rec.model.abstract import RecommenderModule


class ML100KSlateRanker(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int = 32,
        genre_dim: int = 19,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__(project_config, index_mapping)

        self.user_emb = nn.Embedding(self._n_users, n_factors)
        self.item_emb = nn.Embedding(self._n_items, n_factors)

        self.user_tower = nn.Sequential(
            nn.Linear(n_factors, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_factors),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(n_factors + genre_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_factors),
        )

    def forward(self, *args):
        # args order from ProjectConfig.input_columns:
        #   0: user_id          (B,)
        #   1: item_id          (B,)
        #   2: item_genre_vec   (B, 19)  ← from metadata_columns
        user_ids = args[0]
        item_ids = args[1]
        genre_vec = args[2]

        u = self.user_tower(self.user_emb(user_ids))

        item_vec = torch.cat([self.item_emb(item_ids), genre_vec], dim=-1)
        v = self.item_tower(item_vec)

        return (u * v).sum(dim=-1)
