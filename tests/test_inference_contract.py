"""Tests for ``mars_rec.inference``.

The inference pipeline is its own contract: it loads a Lightning
checkpoint, instantiates the recommender module from ``params.json``,
runs preprocessing transforms identical to training, and returns top-k
predictions. Each step has had a silent regression at least once in
the project history — these tests pin them down.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from mars_rec.data.task import BasePrepareDataFrames
from mars_rec.inference import (
    build_reverse_item_mapping,
    load_trained_model,
    predict_top_k,
    preprocess_raw_data,
)
from mars_rec.meta_config import Column, IOType, ProjectConfig
from mars_rec.model.base_model import LogisticRegression


# ---------------------------------------------------------------------------
# A minimal project that the inference module can resolve via load_attr
# ---------------------------------------------------------------------------


class _InfPrepTask(BasePrepareDataFrames):
    """Concrete BasePrepareDataFrames stub so inference can call
    ``transform_data_frame``. We never run() this — only the classmethod-
    style ``transform_data_frame`` is exercised via ``__new__``."""

    @property
    def dataset_dir(self):  # pragma: no cover - never called in these tests
        return "/tmp"

    @property
    def read_data_frame_path(self):  # pragma: no cover
        return None


inference_test_config = ProjectConfig(
    base_dir="/tmp/inf_test",
    prepare_data_frames_task=_InfPrepTask,
    dataset_class=MagicMock(),
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    other_input_columns=[],
    output_column=Column("label", IOType.NUMBER),
)


def _write_task_artifacts(
    task_dir,
    model: torch.nn.Module,
    project_path: str = __name__ + ".inference_test_config",
    module_class_path: str = "mars_rec.model.base_model.LogisticRegression",
    extra_params: dict | None = None,
    write_checkpoint: bool = True,
    state_dict_prefix: str = "module.",
):
    """Persist params.json, index_mapping.pkl, and a Lightning-style checkpoint."""
    if extra_params is None:
        extra_params = {"n_factors": 4}
    params = {
        "project": project_path,
        "recommender_module_class": module_class_path,
        "recommender_extra_params": extra_params,
    }
    with open(task_dir / "params.json", "w") as f:
        json.dump(params, f)

    idx_mapping = {
        "user_id": defaultdict(int, {"u1": 3, "u2": 4}),
        "item_id": defaultdict(int, {"i1": 3, "i2": 4, "i3": 5}),
    }
    with open(task_dir / "index_mapping.pkl", "wb") as f:
        pickle.dump(idx_mapping, f)

    if write_checkpoint:
        # Lightning checkpoint format: {"state_dict": {...}}
        # Keys are prefixed with "module." because the model is wrapped
        # inside a RecommenderLightningModule (self.module = recommender).
        prefixed_sd = {state_dict_prefix + k: v for k, v in model.state_dict().items()}
        torch.save({"state_dict": prefixed_sd}, task_dir / "weights_epoch=00.ckpt")
    return idx_mapping


# ---------------------------------------------------------------------------
# load_trained_model
# ---------------------------------------------------------------------------


class TestLoadTrainedModel:
    def test_returns_tuple_with_expected_types(self, tmp_path):
        # Build the canonical LogisticRegression so its state_dict has real
        # keys matching the constructor.
        idx = {
            "user_id": defaultdict(int, {"u1": 3, "u2": 4}),
            "item_id": defaultdict(int, {"i1": 3, "i2": 4, "i3": 5}),
        }
        model = LogisticRegression(inference_test_config, idx, n_factors=4)
        _write_task_artifacts(tmp_path, model)

        loaded_model, project_config, index_mapping, params = load_trained_model(str(tmp_path))

        assert isinstance(loaded_model, torch.nn.Module)
        assert isinstance(project_config, ProjectConfig)
        assert isinstance(index_mapping, dict)
        assert "user_id" in index_mapping and "item_id" in index_mapping
        assert isinstance(params, dict)
        assert params["recommender_module_class"].endswith("LogisticRegression")

    def test_strips_module_prefix_from_state_dict(self, tmp_path):
        """Lightning saves state_dict with a ``module.`` prefix because the
        recommender lives at ``LightningModule.module``. ``load_trained_model``
        must strip that prefix so the recommender's own keys match."""
        idx = {
            "user_id": defaultdict(int, {"u1": 3, "u2": 4}),
            "item_id": defaultdict(int, {"i1": 3, "i2": 4, "i3": 5}),
        }
        model = LogisticRegression(inference_test_config, idx, n_factors=4)
        # Hard-code known weights so we can assert equality post-load.
        with torch.no_grad():
            model.user_embeddings.weight.fill_(0.42)
            model.linear.bias.fill_(0.7)
        _write_task_artifacts(tmp_path, model, state_dict_prefix="module.")

        loaded_model, *_ = load_trained_model(str(tmp_path))

        assert torch.allclose(
            loaded_model.user_embeddings.weight,
            torch.full_like(loaded_model.user_embeddings.weight, 0.42),
        )
        assert torch.allclose(
            loaded_model.linear.bias,
            torch.tensor([0.7]),
        )

    def test_raises_when_no_checkpoint_present(self, tmp_path):
        idx = {
            "user_id": defaultdict(int, {"u1": 3}),
            "item_id": defaultdict(int, {"i1": 3}),
        }
        model = LogisticRegression(inference_test_config, idx, n_factors=4)
        _write_task_artifacts(tmp_path, model, write_checkpoint=False)

        with pytest.raises(FileNotFoundError, match="No checkpoint"):
            load_trained_model(str(tmp_path))


# ---------------------------------------------------------------------------
# preprocess_raw_data
# ---------------------------------------------------------------------------


class TestPreprocessRawData:
    def test_creates_synthetic_output_column_when_missing(self):
        """Blind test sets may not have the output column. inference.py
        synthesises it from the item column (lines 144-153) so the dataset
        still constructs. This is the silent contract that broke before."""
        idx = {
            "user_id": defaultdict(int, {"u1": 3, "u2": 4}),
            "item_id": defaultdict(int, {"i1": 3, "i2": 4}),
        }
        df_raw = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["i1", "i2"]})
        # Missing: "label" (output column)

        df_out, _ = preprocess_raw_data(df_raw, inference_test_config, idx)
        # output column must now exist
        assert "label" in df_out.columns
        # output column is NUMBER → values should be the placeholder (item id)
        assert len(df_out) == 2

    def test_preserves_output_column_when_present(self):
        idx = {
            "user_id": defaultdict(int, {"u1": 3}),
            "item_id": defaultdict(int, {"i1": 3}),
        }
        df_raw = pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"], "label": [0.8]})
        df_out, _ = preprocess_raw_data(df_raw, inference_test_config, idx)
        # Existing label preserved.
        assert df_out["label"].iloc[0] == pytest.approx(0.8)

    def test_oov_items_map_to_zero(self):
        """Items absent from the training mapping must fall back to the OOV
        index (0) — defaultdict(int) does this — not crash."""
        idx = {
            "user_id": defaultdict(int, {"u1": 3}),
            "item_id": defaultdict(int, {"i_seen": 3}),
        }
        df_raw = pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["i_unseen"],  # OOV
                "label": [1.0],
            }
        )
        df_out, _ = preprocess_raw_data(df_raw, inference_test_config, idx)
        # OOV must be mapped to 0 (the defaultdict default).
        assert df_out["item_id"].iloc[0] == 0


# ---------------------------------------------------------------------------
# predict_top_k
# ---------------------------------------------------------------------------


class TestPredictTopK:
    def test_shape_matches_dataset_size_and_k(self):
        """Build a deterministic dataset + model, run ``predict_top_k`` and
        check the output shape contract: ``[N, k]`` for both indices and
        scores."""

        # We use a fake module that returns logits of shape [B, C] and a
        # synthetic dataset producing tensor batches.
        class _StubDS:
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                arr = np.asarray(idx, dtype=np.int64)
                return (torch.from_numpy(arr),), torch.from_numpy(arr.astype(np.float32))

        class _StubModel(torch.nn.Module):
            def __init__(self, n_classes: int):
                super().__init__()
                self.n_classes = n_classes

            def forward(self, x):
                # x: [B] int — return [B, n_classes] deterministic logits
                B = x.shape[0]
                base = torch.arange(self.n_classes, dtype=torch.float32)
                return base.unsqueeze(0).repeat(B, 1) + x.unsqueeze(1).float() * 0.01

        ds = _StubDS(n=6)
        model = _StubModel(n_classes=10)

        indices, scores = predict_top_k(model, ds, batch_size=3, k=5, device="cpu")
        assert indices.shape == (6, 5)
        assert scores.shape == (6, 5)

    def test_scores_descending_per_row(self):
        class _StubDS:
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

            def __getitem__(self, idx):
                arr = np.asarray(idx, dtype=np.int64)
                return (torch.from_numpy(arr),), torch.from_numpy(arr.astype(np.float32))

        class _StubModel(torch.nn.Module):
            def forward(self, x):
                # Use random but deterministic logits
                torch.manual_seed(0)
                return torch.randn(x.shape[0], 8)

        indices, scores = predict_top_k(_StubModel(), _StubDS(4), batch_size=2, k=3, device="cpu")
        for row in scores:
            # Strictly non-increasing across the top-k axis.
            assert all(row[i] >= row[i + 1] for i in range(len(row) - 1))


# ---------------------------------------------------------------------------
# build_reverse_item_mapping
# ---------------------------------------------------------------------------


class TestBuildReverseItemMapping:
    def test_skips_reserved_indices(self):
        """OOV (0) and PAD (1) must NOT appear in the reverse map — the
        downstream consumer treats them as "no prediction"."""
        idx_mapping = {"item_id": {"oov_marker": 0, np.nan: 1, "-1": 2, "real": 3, "real2": 4}}
        rev = build_reverse_item_mapping(idx_mapping, "item_id")
        # Per inference.py:294, idx <= 1 is skipped; idx 2 (PAD/-1) IS kept.
        assert 0 not in rev
        assert 1 not in rev
        assert rev.get(3) == "real"
        assert rev.get(4) == "real2"
