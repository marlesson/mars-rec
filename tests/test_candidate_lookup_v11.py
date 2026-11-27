"""Unit tests for the v11 derived-feature candidate-lookup overrides.

Tests the three target-aware features added in Phase 3:
  - track_mood_cluster (direct lookup per candidate)
  - seq_isrc_country_match_rate (computed against row's prev_isrc_country)
  - track_mood_cluster_seq_match (computed against row's prev_mood_clusters)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mars_rec.data.dataset import InteractionsDataset
from mars_rec.meta_config import Column, IOType, ProjectConfig, RecommenderType


@pytest.fixture
def project_config_v11():
    return ProjectConfig(
        base_dir="/tmp/v11_test",
        prepare_data_frames_task=None,
        dataset_class=InteractionsDataset,
        dataset_extra_params={},
        user_column=Column("user_id", IOType.INDEXABLE),
        item_column=Column("track_id", IOType.INDEXABLE),
        other_input_columns=[
            Column("prev_isrc_country", IOType.INDEXABLE_ARRAY, same_index_as="track_isrc_country"),
            Column("track_isrc_country", IOType.INDEXABLE),
            Column("prev_mood_clusters", IOType.INDEXABLE_ARRAY, same_index_as="track_mood_cluster"),
            Column("track_mood_cluster", IOType.INDEXABLE),
            Column("seq_isrc_country_match_rate", IOType.NUMBER),
            Column("track_mood_cluster_seq_match", IOType.NUMBER),
        ],
        output_column=Column("label", IOType.NUMBER),
        available_items_column_name=None,
        user_is_input=True,
        item_is_input=True,
        recommender_type=RecommenderType.USER_BASED_COLLABORATIVE_FILTERING,
    )


def _build_dataset(project_config_v11):
    """Minimal dataset with two rows + a candidate lookup."""
    df = pd.DataFrame(
        {
            "user_id": [10, 20],
            "track_id": [100, 200],
            "prev_isrc_country": [[2, 3, 2, 0, 0], [4, 4, 4, 4, 4]],
            "track_isrc_country": [2, 4],
            "prev_mood_clusters": [[5, 5, 6, 0, 0], [7, 7, 7, 8, 0]],
            "track_mood_cluster": [5, 7],
            "seq_isrc_country_match_rate": [0.0, 0.0],
            "track_mood_cluster_seq_match": [0.0, 0.0],
            "label": [1.0, 1.0],
        }
    )
    # Build a fake candidate lookup keyed by item idx 0..3:
    #   idx 1 = track_a (isrc=2, mood=5) — matches row 0 prev_isrc[2,3,2]
    #   idx 2 = track_b (isrc=3, mood=6) — matches row 0 prev_isrc only at pos 1
    #   idx 3 = track_c (isrc=4, mood=7) — matches row 1 perfectly
    #   idx 0 reserved (OOV)
    lookup = {
        "track_isrc_country": np.array([0, 2, 3, 4], dtype=np.int64),
        "track_mood_cluster": np.array([0, 5, 6, 7], dtype=np.int64),
    }
    index_mapping = {
        "user_id": {10: 1, 20: 2},
        "track_id": {100: 1, 200: 2, "track_a": 1, "track_b": 2, "track_c": 3},
        "track_isrc_country": {"PAD": 1, "US": 2, "BR": 3, "JP": 4},
        "track_mood_cluster": {"PAD": 1, "5": 5, "6": 6, "7": 7, "8": 8},
    }
    ds = InteractionsDataset(
        data_frame=df,
        embeddings_for_metadata=None,
        project_config=project_config_v11,
        index_mapping=index_mapping,
        candidate_feature_lookup=lookup,
    )
    return ds


def test_seq_isrc_country_match_rate(project_config_v11):
    ds = _build_dataset(project_config_v11)
    # Build a stub input_tuple matching the column order.
    # Order: user, target, prev_isrc, track_isrc, prev_mood, track_mood, rate1, rate2
    inp = (
        np.array([1, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
        np.array([[2, 3, 2, 0, 0], [4, 4, 4, 4, 4]], dtype=np.int64),
        np.array([2, 4], dtype=np.int64),
        np.array([[5, 5, 6, 0, 0], [7, 7, 7, 8, 0]], dtype=np.int64),
        np.array([5, 7], dtype=np.int64),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    )
    # Apply candidate features for negatives at item idx 1 and 3:
    #   row 0 negative=track_a(idx=1, isrc=2): prev_isrc=[2,3,2,0,0]
    #     valid positions=3 (positions where idx>1: 2,3,2), matches=2 -> 2/3
    #   row 1 negative=track_c(idx=3, isrc=4): prev_isrc=[4,4,4,4,4]
    #     valid=5, matches=5 -> 1.0
    out = ds._apply_candidate_features(inp, np.array([1, 3], dtype=np.int64))
    rate = out[6]
    assert pytest.approx(rate[0], rel=1e-5) == 2.0 / 3.0
    assert pytest.approx(rate[1], rel=1e-5) == 1.0


def test_track_mood_cluster_seq_match(project_config_v11):
    ds = _build_dataset(project_config_v11)
    inp = (
        np.array([1, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
        np.array([[2, 3, 2, 0, 0], [4, 4, 4, 4, 4]], dtype=np.int64),
        np.array([2, 4], dtype=np.int64),
        np.array([[5, 5, 6, 0, 0], [7, 7, 7, 8, 0]], dtype=np.int64),
        np.array([5, 7], dtype=np.int64),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    )
    # neg=track_a(idx=1, mood=5) vs row 0 prev_mood=[5,5,6,0,0]:
    #   valid=3, matches=2 -> 2/3
    # neg=track_c(idx=3, mood=7) vs row 1 prev_mood=[7,7,7,8,0]:
    #   valid=4, matches=3 -> 3/4
    out = ds._apply_candidate_features(inp, np.array([1, 3], dtype=np.int64))
    rate = out[7]
    assert pytest.approx(rate[0], rel=1e-5) == 2.0 / 3.0
    assert pytest.approx(rate[1], rel=1e-5) == 3.0 / 4.0


def test_track_mood_cluster_direct_override(project_config_v11):
    ds = _build_dataset(project_config_v11)
    inp = (
        np.array([1, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
        np.array([[2, 3, 2, 0, 0], [4, 4, 4, 4, 4]], dtype=np.int64),
        np.array([2, 4], dtype=np.int64),
        np.array([[5, 5, 6, 0, 0], [7, 7, 7, 8, 0]], dtype=np.int64),
        np.array([5, 7], dtype=np.int64),  # original positive moods
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    )
    # neg=track_b(idx=2, mood=6), neg=track_a(idx=1, mood=5)
    out = ds._apply_candidate_features(inp, np.array([2, 1], dtype=np.int64))
    cand_mood = out[5]
    assert int(cand_mood[0]) == 6
    assert int(cand_mood[1]) == 5
