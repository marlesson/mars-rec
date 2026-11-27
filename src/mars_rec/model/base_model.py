"""Baseline recommender models shipped with the framework."""

from typing import Callable, Dict, Any

import torch
import torch.nn as nn
from mars_rec.torch.init import lecun_normal_init
from mars_rec.model.abstract import RecommenderModule
from mars_rec.meta_config import ProjectConfig


class LogisticRegression(RecommenderModule):
    """Two-embedding logistic regression baseline.

    Embeds user and item ids into ``n_factors`` dimensions, concatenates,
    runs through a single ``nn.Linear`` head with sigmoid output. Useful
    as a sanity-check baseline against which fancier architectures can
    be compared.

    :param n_factors: Embedding dimensionality (same for user and item).
    :param weight_init: Callable applied to each weight tensor at
        construction time. Defaults to LeCun normal.
    """

    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        weight_init: Callable = lecun_normal_init,
    ):
        super().__init__(project_config, index_mapping)

        self.user_embeddings = nn.Embedding(self._n_users, n_factors)
        self.item_embeddings = nn.Embedding(self._n_items, n_factors)

        weight_init(self.user_embeddings.weight)
        weight_init(self.item_embeddings.weight)

        self.linear = nn.Linear(n_factors * 2, 1)
        self.weight_init = weight_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        """Apply ``self.weight_init`` to every ``nn.Linear``'s weight + bias."""
        if type(module) is nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)

    def forward(self, user_ids, item_ids, context_representation=None):
        """Concatenate user/item embeddings and return a single sigmoid score."""
        x: torch.Tensor = context_representation

        # General embeddings
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        x = torch.cat((user_emb, item_emb), dim=1)

        return torch.sigmoid(self.linear(x))
