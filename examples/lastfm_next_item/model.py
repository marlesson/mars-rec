"""
SASRec-style next-item model with a full-softmax head (trained with ``ce``).

Architecture:
    item_emb (shared by the history and the output head)
    + positional_emb
    -> 1-layer TransformerEncoder (causal mask, masks PAD positions)
    -> last position's hidden state h_u                         (B, d)
    logits = h_u @ item_emb.weight.T                            (B, n_items)

Compared with ``examples/lastfm`` (which scores one sampled candidate per
row with ``bce_logits``), this model scores the whole catalogue in a single
forward pass. That is what ``mars-rec predict`` needs to return a top-k list
per row, and it is the standard objective for sequential recommenders.

The logits of the reserved index slots (unknown / NaN / PAD) stay in the
output; inference masks them before taking the top-k.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from mars_rec.meta_config import ProjectConfig
from mars_rec.model.abstract import RecommenderModule

PAD_INDEX: int = 2  # "-1" maps to index slot 2 (see utils.index_mapping)


class LastFMNextItemSASRec(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int = 64,
        n_heads: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.2,
        history_len: int = 10,
    ):
        super().__init__(project_config, index_mapping)

        self.item_emb = nn.Embedding(self._n_items, n_factors, padding_idx=PAD_INDEX)
        # Small init keeps the initial logits near zero (the default N(0, 1)
        # gives logits of std ~sqrt(n_factors) and a very large starting CE).
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD_INDEX].zero_()
        self.pos_emb = nn.Embedding(history_len, n_factors)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_factors,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        self.out_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(history_len, history_len, dtype=torch.bool), diagonal=1),
        )

    def forward(self, user_ids: torch.Tensor, history_ids: torch.Tensor) -> torch.Tensor:
        # args follow ProjectConfig.input_columns. item_is_input=False, so the
        # target artist is NOT an input:
        #   0: user_id            (B,)  — unused; kept to show the contract
        #   1: history_artist_ids (B, L)
        B, L = history_ids.shape
        positions = torch.arange(L, device=history_ids.device).unsqueeze(0).expand(B, L)
        h = self.item_emb(history_ids) + self.pos_emb(positions)
        h = self.encoder(h, mask=self.causal_mask[:L, :L], src_key_padding_mask=history_ids == PAD_INDEX)
        # History is left-padded, so the last position is always a real item.
        h_user = self.out_dropout(h[:, -1, :])  # (B, d)
        return h_user @ self.item_emb.weight.T  # (B, n_items) logits
