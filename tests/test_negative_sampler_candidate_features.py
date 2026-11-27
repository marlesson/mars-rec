"""Tests for the per-candidate target-feature override hook.

Validates the fix described in
``/Users/marlesson/.claude/plans/tidy-moseying-crescent.md``: every negative
sampling dataset class must replace target-derived features with the
candidate's own metadata (popularity, release_year_bucket, duration_bucket,
album_id, tag_ids, artist_repeat_count, album_continuation), instead of
copying them verbatim from the positive row.

The tests focus on the base ``InteractionsDataset._apply_candidate_features``
helper because it's the single point of truth that all 7 negative-sampling
subclasses call. Verifying the helper directly is faster and more reliable
than retraining or running the full sampling stack — and it also exercises
the lazy-resolution path for the lookup callable.
"""
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mars_rec.data.dataset import InteractionsDataset
from mars_rec.meta_config import Column, IOType, ProjectConfig


def _enhanced_project_config() -> ProjectConfig:
    """Project config that uses a candidate-feature schema
    with only the columns relevant to this test (kept minimal)."""
    return ProjectConfig(
        base_dir="/tmp/test",
        prepare_data_frames_task=MagicMock(),
        dataset_class=MagicMock(),
        user_column=Column("user_id", IOType.INDEXABLE),
        item_column=Column("track_id", IOType.INDEXABLE),
        other_input_columns=[
            Column("prev_artist_ids", IOType.INDEXABLE_ARRAY),
            Column("prev_album_ids", IOType.INDEXABLE_ARRAY, same_index_as="album_id"),
            Column("popularity", IOType.NUMBER),
            Column("release_year_bucket", IOType.INDEXABLE),
            Column("duration_bucket", IOType.INDEXABLE),
            Column("album_id", IOType.INDEXABLE),
            Column("tag_ids", IOType.INT_ARRAY),
            Column("artist_repeat_count", IOType.NUMBER),
            Column("album_continuation", IOType.NUMBER),
        ],
        output_column=Column("label", IOType.NUMBER),
    )


def _empty_df_for_config(cfg: ProjectConfig) -> pd.DataFrame:
    cols = [c.name for c in cfg.input_columns] + [cfg.output_column.name]
    return pd.DataFrame({c: [] for c in cols})


def _index_mapping_for_config() -> dict:
    return {
        "user_id": defaultdict(int, {"1": 1}),
        "track_id": defaultdict(int, {"t1": 1, "t2": 2, "t3": 3, "t4": 4}),
        "prev_artist_ids": defaultdict(int, {"PAD": 1, "a1": 2, "a2": 3, "a3": 4}),
        "album_id": defaultdict(int, {"PAD": 1, "alb1": 2, "alb2": 3, "alb3": 4}),
        "release_year_bucket": defaultdict(int, {"UNK": 1, "2000s": 2, "2010s": 3}),
        "duration_bucket": defaultdict(int, {"UNK": 1, "short": 2, "medium": 3}),
    }


def _synthetic_lookup() -> dict:
    """4-track synthetic lookup keyed by item_index 1..4.

    Track 1: artist=a1, album=alb1, pop=0.1, year=2000s(2), dur=short(2)
    Track 2: artist=a2, album=alb2, pop=0.5, year=2010s(3), dur=medium(3)
    Track 3: artist=a1, album=alb3, pop=0.9, year=2010s(3), dur=short(2)
    Track 4: artist=a3, album=alb1, pop=0.3, year=2000s(2), dur=medium(3)
    """
    n = 5  # indices 0..4 inclusive (0 = OOV)
    return {
        "popularity": np.array([0.0, 0.1, 0.5, 0.9, 0.3], dtype=np.float32),
        "release_year_bucket": np.array([0, 2, 3, 3, 2], dtype=np.int64),
        "duration_bucket": np.array([0, 2, 3, 2, 3], dtype=np.int64),
        "album_id": np.array([1, 2, 3, 4, 2], dtype=np.int64),  # 1=PAD
        "tag_ids": [
            [0, 0, 0],  # idx 0 (OOV) — pad
            [10, 11, 0],  # track 1
            [20, 0, 0],   # track 2
            [10, 30, 31], # track 3
            [40, 0, 0],   # track 4
        ],
        "_artist_per_idx": np.array([1, 2, 3, 2, 4], dtype=np.int64),  # 1=PAD
        "_pad_artist": 1,
        "_pad_album": 1,
    }


@pytest.fixture
def base_dataset() -> InteractionsDataset:
    cfg = _enhanced_project_config()
    df = _empty_df_for_config(cfg)
    return InteractionsDataset(
        df, None, cfg, _index_mapping_for_config(),
        candidate_feature_lookup=_synthetic_lookup(),
    )


class TestApplyCandidateFeaturesNoOp:
    """When no lookup is configured, the helper must return the input
    unchanged so the 12 simple model configs (caser/gru4rec/sasrec/etc.)
    keep their current behavior bit-for-bit."""

    def test_no_op_when_lookup_is_none(self) -> None:
        cfg = _enhanced_project_config()
        df = _empty_df_for_config(cfg)
        ds = InteractionsDataset(df, None, cfg, _index_mapping_for_config())
        # No candidate_feature_lookup kwarg supplied → resolves to None
        original = (
            np.array([1, 2]),  # user_id
            np.array([2, 3]),  # track_id (positive)
            np.array([[2, 3], [3, 4]]),  # prev_artist_ids
            np.array([[2, 3], [4, 1]]),  # prev_album_ids
            np.array([0.1, 0.5], dtype=np.float32),  # popularity (from positive)
            np.array([2, 3]),  # release_year_bucket
            np.array([2, 3]),  # duration_bucket
            np.array([2, 3]),  # album_id
            np.array([[10, 11, 0], [20, 0, 0]]),  # tag_ids
            np.array([1, 0]),  # artist_repeat_count
            np.array([0, 1]),  # album_continuation
        )
        out = ds._apply_candidate_features(original, np.array([3, 4]))
        for a, b in zip(original, out):
            np.testing.assert_array_equal(a, b)


class TestApplyCandidateFeaturesDirectLookups:
    """The five direct-lookup columns must come from the candidate index,
    not from the positive row."""

    def test_popularity_replaced(self, base_dataset) -> None:
        # Positive row has popularity 0.1 (track 1); negative is track 3
        positive_input = (
            np.array([1, 1]),                          # user_id
            np.array([1, 1]),                          # track_id (will be overwritten)
            np.array([[2, 1], [2, 1]]),                # prev_artist_ids
            np.array([[2, 1], [2, 1]]),                # prev_album_ids
            np.array([0.1, 0.1], dtype=np.float32),    # popularity (POSITIVE's)
            np.array([2, 2]),                          # release_year_bucket
            np.array([2, 2]),                          # duration_bucket
            np.array([2, 2]),                          # album_id
            np.array([[10, 11, 0], [10, 11, 0]]),      # tag_ids
            np.array([0, 0]),                          # artist_repeat_count
            np.array([0, 0]),                          # album_continuation
        )
        out = base_dataset._apply_candidate_features(
            positive_input, np.array([3, 4])
        )
        # popularity column should now reflect tracks 3 and 4
        np.testing.assert_array_almost_equal(out[4], np.array([0.9, 0.3]))

    def test_release_year_duration_album_replaced(self, base_dataset) -> None:
        positive_input = (
            np.array([1]), np.array([1]),
            np.array([[2, 1]]), np.array([[2, 1]]),
            np.array([0.1], dtype=np.float32),
            np.array([2]),  # year (POSITIVE: 2000s)
            np.array([2]),  # duration (POSITIVE: short)
            np.array([2]),  # album (POSITIVE: alb1)
            np.array([[10, 11, 0]]),
            np.array([0]), np.array([0]),
        )
        # Negative is track 2 (year=2010s=3, duration=medium=3, album=alb2=3)
        out = base_dataset._apply_candidate_features(positive_input, np.array([2]))
        np.testing.assert_array_equal(out[5], [3])  # release_year_bucket
        np.testing.assert_array_equal(out[6], [3])  # duration_bucket
        np.testing.assert_array_equal(out[7], [3])  # album_id

    def test_tag_ids_replaced(self, base_dataset) -> None:
        positive_input = (
            np.array([1]), np.array([1]),
            np.array([[2, 1]]), np.array([[2, 1]]),
            np.array([0.1], dtype=np.float32),
            np.array([2]), np.array([2]), np.array([2]),
            np.array([[10, 11, 0]]),  # POSITIVE's tags
            np.array([0]), np.array([0]),
        )
        out = base_dataset._apply_candidate_features(positive_input, np.array([3]))
        # Track 3 tags = [10, 30, 31]
        np.testing.assert_array_equal(out[8][0], [10, 30, 31])


class TestApplyCandidateFeaturesComputed:
    """``artist_repeat_count`` and ``album_continuation`` must be recomputed
    against the candidate's own artist/album, not stay at the positive's."""

    def test_artist_repeat_count_zero_when_artist_absent(
        self, base_dataset,
    ) -> None:
        # prev_artist_ids = [a2(3), a3(4)]; candidate track 1 has artist a1(2)
        positive_input = (
            np.array([1]), np.array([1]),
            np.array([[3, 4]]),  # prev_artist_ids
            np.array([[3, 4]]),
            np.array([0.5], dtype=np.float32),
            np.array([2]), np.array([2]), np.array([2]),
            np.array([[0, 0, 0]]),
            np.array([99]),  # POSITIVE's value (should be overwritten)
            np.array([0]),
        )
        out = base_dataset._apply_candidate_features(positive_input, np.array([1]))
        np.testing.assert_array_equal(out[9], [0])

    def test_artist_repeat_count_counts_matches(self, base_dataset) -> None:
        # prev_artist_ids = [a1(2), a1(2), a2(3)]; candidate track 1 = artist a1
        positive_input = (
            np.array([1]), np.array([1]),
            np.array([[2, 2, 3]]),  # prev_artist_ids: a1, a1, a2
            np.array([[2, 2, 3]]),
            np.array([0.5], dtype=np.float32),
            np.array([2]), np.array([2]), np.array([2]),
            np.array([[0, 0, 0]]),
            np.array([0]), np.array([0]),
        )
        out = base_dataset._apply_candidate_features(positive_input, np.array([1]))
        # Candidate track 1 has artist a1(2), appears twice in prev → 2
        np.testing.assert_array_equal(out[9], [2])

    def test_album_continuation_one_when_last_album_matches(
        self, base_dataset,
    ) -> None:
        # prev_album_ids ends in alb2(3); candidate track 2 has album alb2(3)
        positive_input = (
            np.array([1]), np.array([1]),
            np.array([[2, 1]]),
            np.array([[1, 3]]),  # prev_album_ids: PAD then alb2
            np.array([0.5], dtype=np.float32),
            np.array([2]), np.array([2]), np.array([2]),
            np.array([[0, 0, 0]]),
            np.array([0]),
            np.array([0]),
        )
        out = base_dataset._apply_candidate_features(positive_input, np.array([2]))
        np.testing.assert_array_equal(out[10], [1])

    def test_album_continuation_zero_when_last_album_differs(
        self, base_dataset,
    ) -> None:
        # prev ends alb3(4); candidate is track 1 with album alb1(2) — no match
        positive_input = (
            np.array([1]), np.array([1]),
            np.array([[2, 1]]),
            np.array([[1, 4]]),  # prev: PAD, alb3
            np.array([0.5], dtype=np.float32),
            np.array([2]), np.array([2]), np.array([2]),
            np.array([[0, 0, 0]]),
            np.array([0]), np.array([0]),
        )
        out = base_dataset._apply_candidate_features(positive_input, np.array([1]))
        np.testing.assert_array_equal(out[10], [0])


class TestLookupResolution:
    """Validate the deferred-callable resolution path that configs use."""

    def test_zero_arg_callable_resolved(self) -> None:
        cfg = _enhanced_project_config()
        df = _empty_df_for_config(cfg)
        called = {"count": 0}
        target = _synthetic_lookup()

        def factory():
            called["count"] += 1
            return target

        ds = InteractionsDataset(
            df, None, cfg, _index_mapping_for_config(),
            candidate_feature_lookup=factory,
        )
        out1 = ds._resolve_candidate_lookup()
        out2 = ds._resolve_candidate_lookup()  # cached — should not call again
        assert out1 is target
        assert out2 is target
        assert called["count"] == 1

    def test_one_arg_callable_receives_index_mapping(self) -> None:
        cfg = _enhanced_project_config()
        df = _empty_df_for_config(cfg)
        captured = {}
        target = _synthetic_lookup()

        def factory(index_mapping):
            captured["index_mapping"] = index_mapping
            return target

        idx_map = _index_mapping_for_config()
        ds = InteractionsDataset(
            df, None, cfg, idx_map, candidate_feature_lookup=factory,
        )
        out = ds._resolve_candidate_lookup()
        assert out is target
        assert captured["index_mapping"] is idx_map

    def test_dict_passed_through_unchanged(self) -> None:
        cfg = _enhanced_project_config()
        df = _empty_df_for_config(cfg)
        target = _synthetic_lookup()
        ds = InteractionsDataset(
            df, None, cfg, _index_mapping_for_config(),
            candidate_feature_lookup=target,
        )
        assert ds._resolve_candidate_lookup() is target

    def test_none_returns_none(self) -> None:
        cfg = _enhanced_project_config()
        df = _empty_df_for_config(cfg)
        ds = InteractionsDataset(df, None, cfg, _index_mapping_for_config())
        assert ds._resolve_candidate_lookup() is None
