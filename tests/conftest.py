"""Shared fixtures for the mars-rec test suite.

These fixtures cover the small set of objects that every framework-level
test needs: a minimal ``ProjectConfig``, an ``index_mapping`` honouring
the PAD=1/-1=2 reserved indices, a small interaction DataFrame, and an
``nn.Module`` stub for Lightning/optimizer plumbing tests.

Tests that need richer schemas (per-candidate features, metadata
embeddings, sequence inputs) are expected to extend these locally rather
than pollute conftest with project-specific defaults.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from mars_rec.data.dataset import InteractionsDataset
from mars_rec.meta_config import Column, IOType, ProjectConfig


# ---------------------------------------------------------------------------
# ProjectConfig fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_project_config() -> ProjectConfig:
    """Two indexable columns (user/item) + a NUMBER label.

    Matches the contract LogisticRegression expects:
    ``forward(user_ids, item_ids, context_representation=None) -> [B, 1]``.
    ``output_column`` is a binary 0/1 NUMBER (BCE-style).
    """
    return ProjectConfig(
        base_dir="/tmp/test",
        prepare_data_frames_task=MagicMock(),
        dataset_class=InteractionsDataset,
        user_column=Column("user_id", IOType.INDEXABLE),
        item_column=Column("item_id", IOType.INDEXABLE),
        other_input_columns=[],
        output_column=Column("label", IOType.NUMBER),
    )


@pytest.fixture
def mixed_project_config() -> ProjectConfig:
    """user + item + 1 NUMBER + 1 INDEXABLE_ARRAY input — covers tuple unpack."""
    return ProjectConfig(
        base_dir="/tmp/test",
        prepare_data_frames_task=MagicMock(),
        dataset_class=InteractionsDataset,
        user_column=Column("user_id", IOType.INDEXABLE),
        item_column=Column("item_id", IOType.INDEXABLE),
        other_input_columns=[
            Column("feat_num", IOType.NUMBER),
            Column("prev_items", IOType.INDEXABLE_ARRAY, same_index_as="item_id"),
        ],
        output_column=Column("label", IOType.NUMBER),
    )


# ---------------------------------------------------------------------------
# Index mapping fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def index_mapping_small() -> Dict[str, Dict[Any, int]]:
    """Honours framework convention: 0=OOV, 1=NaN, 2=-1/PAD, real values start at 3.

    create_index_mapping in src/mars_rec/utils/index_mapping.py reserves
    these slots. The fixture mirrors that so tests of OOV/PAD fallback work
    the same way the real pipeline does.
    """
    user_mapping = defaultdict(int, {"u1": 3, "u2": 4, "u3": 5})
    item_mapping = defaultdict(int, {"i1": 3, "i2": 4, "i3": 5, "i4": 6})
    return {"user_id": user_mapping, "item_id": item_mapping}


# ---------------------------------------------------------------------------
# Data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_interactions_df() -> pd.DataFrame:
    """30 rows = 3 users × 10 items. Indexed values (already mapped to ints)."""
    rng = np.random.default_rng(seed=42)
    rows = []
    user_ids = [3, 4, 5]  # matches index_mapping_small
    item_ids = [3, 4, 5, 6]
    for u in user_ids:
        for _ in range(10):
            i = int(rng.choice(item_ids))
            label = float(rng.random() > 0.5)
            rows.append({"user_id": u, "item_id": i, "label": label})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_task_dir(tmp_path):
    """A scratch directory that looks like a Luigi task output dir.

    Used by inference-loading tests that need to point ``load_trained_model``
    at a directory containing ``params.json``, ``weights*.ckpt``, and
    ``index_mapping.pkl``.
    """
    d = tmp_path / "task_output"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Tiny nn.Module fixture (for Lightning / optimizer tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_torch_module() -> nn.Module:
    """Linear 4→1 model with sigmoid — gradients flow, params are trainable."""
    class _Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    torch.manual_seed(0)
    return _Dummy()
