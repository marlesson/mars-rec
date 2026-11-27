"""
SASRec-lite next-item ranking model for the Last.fm sample.

Architecture:
    item_emb (shared between history + target)
    + positional_emb
    -> 1-layer TransformerEncoder (causal mask, masks PAD positions)
    -> take last non-PAD position's hidden state h_u
    score = <h_u, item_emb(target)>

This is the pedagogical example for sequential recommendation in
mars-rec: it shows how a sequence column declared as
``IOType.INDEXABLE_ARRAY`` with ``same_index_as=item_column`` is consumed
end-to-end with a single shared item embedding.
"""
from typing import Any, Dict

import torch
import torch.nn as nn

from mars_rec.meta_config import ProjectConfig
from mars_rec.model.abstract import RecommenderModule

# Index value used to pad the history sequence. ``create_index_mapping`` in
# ``utils.index_mapping`` maps the string ``"-1"`` to slot 2, regardless of
# vocabulary contents.
PAD_INDEX: int = 2


class LastFMSASRec(RecommenderModule):
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

        # ``_n_items`` is the size of the artist vocabulary, set by the base
        # ``RecommenderModule`` from the index_mapping. history_artist_ids
        # shares this vocabulary (via ``same_index_as``).
        self.item_emb = nn.Embedding(
            self._n_items, n_factors, padding_idx=PAD_INDEX,
        )
        self.pos_emb = nn.Embedding(history_len, n_factors)
        self.history_len = history_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_factors,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.out_dropout = nn.Dropout(dropout)

        # Causal mask: position i can only attend to positions <= i.
        causal = torch.triu(
            torch.ones(history_len, history_len, dtype=torch.bool), diagonal=1,
        )
        self.register_buffer("causal_mask", causal)

    def forward(self, *args):
        # args order follows ProjectConfig.input_columns:
        #   0: user_id        (B,)
        #   1: artist_id      (B,)  — target item
        #   2: history_artist_ids (B, L)
        target_ids = args[1]
        history_ids = args[2]

        B, L = history_ids.shape
        pad_mask = history_ids == PAD_INDEX  # (B, L), True where padded

        positions = torch.arange(L, device=history_ids.device).unsqueeze(0).expand(B, L)
        h = self.item_emb(history_ids) + self.pos_emb(positions)
        h = self.encoder(
            h,
            mask=self.causal_mask[:L, :L],
            src_key_padding_mask=pad_mask,
        )
        h = self.out_dropout(h)

        # Pool: take the last non-PAD position per row. Because we left-pad in
        # data.py the last position is guaranteed to be non-PAD when the user
        # has any history at all, so a plain ``h[:, -1]`` suffices.
        h_user = h[:, -1, :]  # (B, d)

        target_vec = self.item_emb(target_ids)  # (B, d)
        return (h_user * target_vec).sum(dim=-1)
