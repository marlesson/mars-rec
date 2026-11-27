"""
Neural network models for MovieLens 1M recommendation.
"""
from typing import Any, Dict

import torch
import torch.nn as nn

from mars_rec.meta_config import ProjectConfig
from mars_rec.model.abstract import RecommenderModule


class MovieLensNCF(RecommenderModule):
    """Neural Collaborative Filtering model for MovieLens.

    Combines user/item embeddings (GMF path) with an MLP path that
    also incorporates user demographic features and movie genre features.

    Architecture:
        GMF: user_emb * item_emb (element-wise product)
        MLP: concat(user_emb, item_emb, user_features, genre_features) -> MLP
        Output: concat(GMF, MLP) -> Linear -> sigmoid
    """

    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int = 32,
        hidden_size: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__(project_config, index_mapping)

        # GMF path embeddings
        self.user_emb_gmf = nn.Embedding(self._n_users, n_factors)
        self.item_emb_gmf = nn.Embedding(self._n_items, n_factors)

        # MLP path embeddings
        self.user_emb_mlp = nn.Embedding(self._n_users, n_factors)
        self.item_emb_mlp = nn.Embedding(self._n_items, n_factors)

        # Number of extra features: 3 user features + 18 genre features
        n_extra = len(project_config.other_input_columns)
        mlp_input = 2 * n_factors + n_extra

        # Build MLP layers
        layers = []
        in_size = mlp_input
        for i in range(n_layers):
            out_size = hidden_size // (2 ** i)
            layers.extend([
                nn.Linear(in_size, out_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_size = out_size
        self.mlp = nn.Sequential(*layers)

        # Final prediction: GMF (n_factors) + MLP output
        self.output_layer = nn.Linear(n_factors + in_size, 1)

    def forward(self, *args):
        # args order follows ProjectConfig.input_columns:
        # user_ids, item_ids, gender, age, occupation, 18 genre features
        user_ids = args[0]
        item_ids = args[1]

        # GMF path
        gmf = self.user_emb_gmf(user_ids) * self.item_emb_gmf(item_ids)

        # MLP path
        user_mlp = self.user_emb_mlp(user_ids)
        item_mlp = self.item_emb_mlp(item_ids)

        # Stack all extra features (3 user + 18 genre = 21)
        extra_features = torch.stack(list(args[2:]), dim=1)

        mlp_input = torch.cat([user_mlp, item_mlp, extra_features], dim=1)
        mlp_out = self.mlp(mlp_input)

        # Combine GMF + MLP
        combined = torch.cat([gmf, mlp_out], dim=1)
        return self.output_layer(combined).squeeze(-1)
