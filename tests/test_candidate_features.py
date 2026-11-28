"""Tests for per-candidate feature recomputation (``mars_rec.data.candidate_features``).

The schema is synthetic: each item has a ``category`` and a ``score``; each
row has a ``hist_categories`` history and a derived ``category_in_history``
count that a project hook recomputes for the candidate. The same helper runs
in the negative samplers, the training-time evaluation and inference, and the
tests check each of those paths plus their parity.
"""

from __future__ import annotations

import types
import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from torch import nn

from mars_rec.data import candidate_features as cf
from mars_rec.data.candidate_features import (
    apply_candidate_features,
    apply_candidate_features_to_frame,
    convert_dtype,
    resolve_candidate_lookup,
    validate_candidate_feature_fn,
)
from mars_rec.data.dataset import InteractionsDataset, InteractionsWithNegativeItemGenerationDataset
from mars_rec.meta_config import Column, IOType, ProjectConfig
from mars_rec.utils.index_mapping import create_index_mapping

N_ITEMS = 7  # indices 0..2 reserved, 3..6 real

# Per-item metadata keyed by item index.
CATEGORY = np.array([0, 0, 0, 10, 11, 10, 12], dtype=np.int64)
SCORE = np.array([0.0, 0.0, 0.0, 0.3, 0.6, 0.9, 0.1], dtype=np.float32)
ATTRS = [[0, 0], [0, 0], [0, 0], [1, 2], [3, 0], [4, 5], [6, 0]]  # ragged-style list of lists


def _config(dataset_extra_params=None) -> ProjectConfig:
    return ProjectConfig(
        base_dir="/tmp/test",
        prepare_data_frames_task=MagicMock(),
        dataset_class=MagicMock(),
        dataset_extra_params=dataset_extra_params or {},
        user_column=Column("user_id", IOType.INDEXABLE),
        item_column=Column("item_id", IOType.INDEXABLE),
        other_input_columns=[
            Column("hist_categories", IOType.INT_ARRAY),
            Column("category", IOType.INDEXABLE),
            Column("score", IOType.NUMBER),
            Column("attrs", IOType.INT_ARRAY),
            Column("category_in_history", IOType.NUMBER),
        ],
        output_column=Column("label", IOType.NUMBER),
    )


def _lookup() -> dict:
    return {
        "category": CATEGORY,
        "score": SCORE,
        "attrs": ATTRS,
        "_category_pad": 0,  # private: hook-only data, never a column
    }


def category_in_history(columns, candidate_idx, lookup):
    """Synthetic project hook: count the candidate's category in the row history."""
    hist = np.asarray(columns["hist_categories"], dtype=np.int64)  # [B, L]
    cand_cat = columns["category"]  # direct lookup already applied
    valid = hist != lookup["_category_pad"]
    return {"category_in_history": ((hist == cand_cat[:, None]) & valid).sum(axis=1)}


def _columns_batch() -> dict:
    """Two rows whose positives are items 3 and 4."""
    return {
        "user_id": np.array([3, 4], dtype=np.int64),
        "item_id": np.array([3, 4], dtype=np.int64),
        "hist_categories": np.array([[10, 10, 0], [12, 11, 0]], dtype=np.int64),
        "category": np.array([10, 11], dtype=np.int64),
        "score": np.array([0.3, 0.6], dtype=np.float32),
        "attrs": np.array([[1, 2], [3, 0]], dtype=np.int64),
        "category_in_history": np.array([2.0, 1.0], dtype=np.float32),
    }


@pytest.fixture(autouse=True)
def _reset_warn_once(monkeypatch):
    monkeypatch.setattr(cf, "_warned_private_keys", False)


# ---------------------------------------------------------------------------
# resolve / validate
# ---------------------------------------------------------------------------


class TestResolveLookup:
    def test_none(self):
        assert resolve_candidate_lookup(None, {}) is None

    def test_dict_passthrough(self):
        lk = _lookup()
        assert resolve_candidate_lookup(lk, {}) is lk

    def test_zero_arg_callable(self):
        lk = _lookup()
        assert resolve_candidate_lookup(lambda: lk, {}) is lk

    def test_one_arg_callable_receives_index_mapping(self):
        seen = {}
        idx = {"item_id": {}}

        def factory(index_mapping):
            seen["idx"] = index_mapping
            return {}

        resolve_candidate_lookup(factory, idx)
        assert seen["idx"] is idx

    def test_rejects_non_dict_non_callable(self):
        with pytest.raises(TypeError, match="dict or callable"):
            resolve_candidate_lookup([1, 2], {})

    def test_rejects_callable_returning_non_dict(self):
        with pytest.raises(TypeError, match="must return a dict"):
            resolve_candidate_lookup(lambda: [1], {})

    def test_validate_fn(self):
        assert validate_candidate_feature_fn(None) is None
        assert validate_candidate_feature_fn(category_in_history) is category_in_history
        with pytest.raises(TypeError, match="must be callable"):
            validate_candidate_feature_fn("not callable")


# ---------------------------------------------------------------------------
# apply_candidate_features
# ---------------------------------------------------------------------------


class TestApplyCandidateFeatures:
    def test_noop_without_lookup_or_fn(self):
        assert apply_candidate_features(_columns_batch(), np.array([5, 6]), _config().input_columns, None) == {}

    def test_direct_lookups_follow_lookup_keys_and_cast_dtype(self):
        lk = {k: v for k, v in _lookup().items() if not k.startswith("_")}
        out = apply_candidate_features(_columns_batch(), np.array([5, 6]), _config().input_columns, lk)
        assert set(out) == {"category", "score", "attrs"}
        np.testing.assert_array_equal(out["category"], [10, 12])
        assert out["category"].dtype == np.int64
        np.testing.assert_allclose(out["score"], [0.9, 0.1])
        assert out["score"].dtype == np.float32
        np.testing.assert_array_equal(out["attrs"], [[4, 5], [6, 0]])
        assert out["attrs"].dtype == np.int64

    def test_columns_absent_from_lookup_are_untouched(self):
        lk = {"score": SCORE}
        out = apply_candidate_features(_columns_batch(), np.array([5, 6]), _config().input_columns, lk)
        assert set(out) == {"score"}

    def test_item_column_and_private_keys_never_written(self):
        lk = {**_lookup(), "item_id": np.arange(N_ITEMS) + 100}
        with pytest.warns(UserWarning, match="private keys"):
            out = apply_candidate_features(
                _columns_batch(), np.array([5, 6]), _config().input_columns, lk, item_column_name="item_id"
            )
        assert "item_id" not in out and "_category_pad" not in out

    def test_hook_sees_direct_lookups_and_recomputes(self):
        out = apply_candidate_features(
            _columns_batch(), np.array([5, 6]), _config().input_columns, _lookup(), category_in_history
        )
        # row 0: candidate 5 → category 10, history [10, 10, pad] → 2
        # row 1: candidate 6 → category 12, history [12, 11, pad] → 1
        np.testing.assert_array_equal(out["category_in_history"], [2.0, 1.0])
        assert out["category_in_history"].dtype == np.float32

    def test_hook_without_lookup(self):
        def fn(columns, cand, lookup):
            assert lookup == {}
            return {"score": cand * 0.5}

        out = apply_candidate_features(_columns_batch(), np.array([4, 6]), _config().input_columns, None, fn)
        np.testing.assert_allclose(out["score"], [2.0, 3.0])

    def test_hook_unknown_column_raises(self):
        with pytest.raises(KeyError, match="not an input column"):
            apply_candidate_features(
                _columns_batch(), np.array([5, 6]), _config().input_columns, None, lambda c, i, lk: {"nope": i}
            )

    def test_hook_cannot_overwrite_item_column(self):
        with pytest.raises(KeyError, match="item column"):
            apply_candidate_features(
                _columns_batch(),
                np.array([5, 6]),
                _config().input_columns,
                None,
                lambda c, i, lk: {"item_id": i},
                item_column_name="item_id",
            )

    def test_private_keys_without_hook_warn_once(self):
        cols = _config().input_columns
        with pytest.warns(UserWarning, match="candidate_feature_fn"):
            apply_candidate_features(_columns_batch(), np.array([5, 6]), cols, _lookup())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            apply_candidate_features(_columns_batch(), np.array([5, 6]), cols, _lookup())

    def test_no_warning_with_hook(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            apply_candidate_features(
                _columns_batch(), np.array([5, 6]), _config().input_columns, _lookup(), category_in_history
            )


class TestApplyToFrame:
    def _frame(self) -> pd.DataFrame:
        b = _columns_batch()
        return pd.DataFrame({k: list(v) if v.ndim > 1 else v for k, v in b.items()})

    def test_noop_returns_same_frame(self):
        df = self._frame()
        assert apply_candidate_features_to_frame(df, np.array([5, 6]), _config().input_columns, None) is df

    def test_rewrites_scalar_and_array_columns(self):
        df = apply_candidate_features_to_frame(
            self._frame(), np.array([5, 6]), _config().input_columns, _lookup(), category_in_history, "item_id"
        )
        np.testing.assert_array_equal(df["category"], [10, 12])
        np.testing.assert_allclose(df["score"], [0.9, 0.1])
        assert [list(a) for a in df["attrs"]] == [[4, 5], [6, 0]]
        np.testing.assert_array_equal(df["category_in_history"], [2.0, 1.0])

    def test_uncastable_column_passed_through(self):
        df = self._frame()
        df["user_id"] = ["raw-a", "raw-b"]  # INDEXABLE but not castable to int
        out = apply_candidate_features_to_frame(df, np.array([5, 6]), _config().input_columns, {"score": SCORE})
        assert list(out["user_id"]) == ["raw-a", "raw-b"]

    def test_convert_dtype_ragged_arrays(self):
        out = convert_dtype([[1, 2], [3]], IOType.INT_ARRAY)
        assert [list(a) for a in out] == [[1, 2], [3]]
        out = convert_dtype([[0.5], [1.0, 2.0]], IOType.FLOAT_ARRAY)
        assert out[1].dtype == np.float32


# ---------------------------------------------------------------------------
# Negative samplers (train time)
# ---------------------------------------------------------------------------


def _index_mapping() -> dict:
    return {
        "user_id": create_index_mapping(["u1", "u2"]),
        "item_id": create_index_mapping(list("abcd")),  # → 3..6
    }


def _train_df() -> pd.DataFrame:
    b = _columns_batch()
    return pd.DataFrame({**{k: list(v) if v.ndim > 1 else v for k, v in b.items()}, "label": [1.0, 1.0]})


class TestDatasetIntegration:
    def test_base_dataset_noop_without_params(self):
        ds = InteractionsDataset(_train_df(), None, _config(), _index_mapping())
        inp, _ = ds[[0, 1]]
        out = ds._apply_candidate_features(inp, np.array([5, 6]))
        for a, b in zip(inp, out):
            np.testing.assert_array_equal(a, b)

    def test_rejects_non_callable_fn(self):
        with pytest.raises(TypeError, match="must be callable"):
            InteractionsDataset(_train_df(), None, _config(), _index_mapping(), candidate_feature_fn=1)

    def test_lookup_callable_resolved_once(self):
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return _lookup()

        ds = InteractionsDataset(
            _train_df(),
            None,
            _config(),
            _index_mapping(),
            candidate_feature_lookup=factory,
            candidate_feature_fn=category_in_history,
        )
        ds._resolve_candidate_lookup()
        ds._resolve_candidate_lookup()
        assert calls["n"] == 1

    def test_negatives_carry_candidate_features(self):
        cfg = _config()
        ds = InteractionsWithNegativeItemGenerationDataset(
            _train_df(),
            None,
            cfg,
            _index_mapping(),
            negative_proportion=0.5,
            candidate_feature_lookup=_lookup(),
            candidate_feature_fn=category_in_history,
        )
        names = [c.name for c in cfg.input_columns]
        for _ in range(20):
            inp, out = ds[list(range(len(ds)))]
            cols = dict(zip(names, inp))
            neg = out == 0.0
            items = cols["item_id"][neg]
            np.testing.assert_array_equal(cols["category"][neg], CATEGORY[items])
            np.testing.assert_allclose(cols["score"][neg], SCORE[items])
            np.testing.assert_array_equal(cols["attrs"][neg], np.asarray(ATTRS)[items])
            hist = cols["hist_categories"][neg]
            expected = ((hist == CATEGORY[items][:, None]) & (hist != 0)).sum(axis=1)
            np.testing.assert_array_equal(cols["category_in_history"][neg], expected)
            # dtypes stay aligned with the positive half (no float64 promotion)
            assert cols["score"].dtype == np.float32
            assert cols["category_in_history"].dtype == np.float32


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class _EchoInput(nn.Module):
    """Returns one input column as the score, so tests can read what the model saw."""

    def __init__(self, slot: int):
        super().__init__()
        self.slot = slot

    def forward(self, *features):
        return features[self.slot].float()


class TestInferenceDefault:
    def _score(self, cfg, slot, **kwargs):
        from mars_rec.inference import score_candidates_bce

        df = _train_df().iloc[:1].reset_index(drop=True)
        scores, eligible = score_candidates_bce(
            _EchoInput(slot),
            df,
            cfg,
            _index_mapping(),
            np.array([3, 4, 5, 6], dtype=np.int64),
            show_progress=False,
            **kwargs,
        )
        return scores[0].numpy(), eligible[0]

    def test_applies_project_params_by_default(self):
        cfg = _config({"candidate_feature_lookup": _lookup(), "candidate_feature_fn": category_in_history})
        names = [c.name for c in cfg.input_columns]
        s, elig = self._score(cfg, names.index("score"))
        np.testing.assert_allclose(s, SCORE[elig])
        s, elig = self._score(cfg, names.index("category_in_history"))
        np.testing.assert_array_equal(s, [2, 0, 2, 0])

    def test_explicit_override_replaces_default(self):
        cfg = _config({"candidate_feature_lookup": _lookup()})
        names = [c.name for c in cfg.input_columns]

        def override(df, cand, offsets):
            df["score"] = -1.0
            return df

        s, _ = self._score(cfg, names.index("score"), candidate_feature_overrides=override)
        np.testing.assert_array_equal(s, [-1.0] * 4)

    def test_noop_without_params(self):
        cfg = _config()
        names = [c.name for c in cfg.input_columns]
        s, _ = self._score(cfg, names.index("score"))
        np.testing.assert_allclose(s, [0.3] * 4, rtol=1e-6)
