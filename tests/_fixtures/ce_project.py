"""A minimal next-item (CE) project, importable via dotted path.

Used by the inference / evaluation tests that exercise the
``[batch, n_items]`` logits path (``predict_top_k``, ``run_inference``,
``inference.run_evaluate``, ``score_candidates_ce`` / ``_emb``), which the
point-wise ``dummy_project`` cannot reach.

The model is an "oracle": a ``[n_users, n_items]`` logit table that the
tests pin so each user's ground-truth item is ranked first. That turns
ranking metrics into exact, hand-checkable values.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mars_rec.data.dataset import InteractionsDataset
from mars_rec.data.task import BasePrepareDataFrames
from mars_rec.meta_config import Column, IOType, ProjectConfig
from mars_rec.model.abstract import RecommenderModule


class CEPrep(BasePrepareDataFrames):
    @property
    def dataset_dir(self) -> str:
        return os.environ.get("MARS_TEST_DATASET_DIR", "/tmp/mars_test_ce_dataset")

    @property
    def read_data_frame_path(self):  # pragma: no cover - never read
        return None


ce_project_config = ProjectConfig(
    base_dir="/tmp/mars_test_ce_dataset",
    prepare_data_frames_task=CEPrep,
    dataset_class=InteractionsDataset,
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    item_is_input=False,
    other_input_columns=[],
    # The target *is* the item: logits index == item index.
    output_column=Column("item_id", IOType.INDEXABLE),
)


class OracleCE(RecommenderModule):
    """``forward(user_ids) -> [B, n_items]`` logits from a lookup table."""

    def __init__(self, project_config, index_mapping, **kwargs):
        super().__init__(project_config, index_mapping)
        self.table = nn.Embedding(self._n_users, self._n_items)
        nn.init.zeros_(self.table.weight)

    def forward(self, user_ids):
        return self.table(user_ids)

    # Embedding-scoring interface used by ``score_candidates_emb``: the
    # "item embedding" is a one-hot row, so ``pred @ item.T`` recovers the
    # (normalised) logit of each candidate.
    def get_target_emb(self, item_ids):
        return F.one_hot(item_ids, num_classes=self._n_items).float()


def pin_oracle(model: OracleCE, user_to_item: dict) -> None:
    """Give ``user_idx -> item_idx`` a high logit and every other item a
    strictly decreasing one, so the full ranking is deterministic."""
    with torch.no_grad():
        n_items = model.table.weight.shape[1]
        base = -torch.arange(n_items, dtype=torch.float32) / n_items
        model.table.weight.copy_(base.expand_as(model.table.weight))
        for u, i in user_to_item.items():
            model.table.weight[u, i] = 10.0
