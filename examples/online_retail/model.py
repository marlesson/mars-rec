"""
GRU4Rec-style session-based ranking model.

Reads the per-session history of stock codes via a single shared item
embedding, runs a 1-layer GRU over the sequence, and scores the candidate
item by dot product against the final hidden state.

This is the pedagogical counterpart to the attention-based SASRec-lite of
the Last.fm sample: same recipe (shared item emb + sequence encoder +
dot-product score) but with recurrence instead of self-attention, and with
the user_column bound to an ephemeral session id.
"""

from typing import Any, Dict

import torch.nn as nn

from mars_rec.meta_config import ProjectConfig
from mars_rec.model.abstract import RecommenderModule

PAD_INDEX: int = 2


class OnlineRetailGRU4Rec(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int = 64,
        hidden_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__(project_config, index_mapping)

        self.item_emb = nn.Embedding(
            self._n_items,
            n_factors,
            padding_idx=PAD_INDEX,
        )
        self.gru = nn.GRU(
            input_size=n_factors,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, n_factors) if hidden_size != n_factors else nn.Identity()

    def forward(self, *args):
        # args order from ProjectConfig.input_columns:
        #   0: session_id          (B,)
        #   1: stock_code          (B,)  — target item
        #   2: history_stock_codes (B, L)
        target_ids = args[1]
        history_ids = args[2]

        # Build a length per row from the PAD positions so the final hidden
        # state reflects only the real items, not padding. History is
        # LEFT-padded, so the last position is always real for non-empty
        # histories.
        h_seq = self.item_emb(history_ids)
        h_seq = self.dropout(h_seq)
        out, _ = self.gru(h_seq)  # (B, L, H)
        h_session = self.proj(out[:, -1, :])  # (B, d)

        target_vec = self.item_emb(target_ids)  # (B, d)
        return (h_session * target_vec).sum(dim=-1)
