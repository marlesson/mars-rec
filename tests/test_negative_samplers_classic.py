"""Tests for the classic negative-sampling datasets in ``mars_rec.data.dataset``.

Covers:
- InteractionsWithNegativeItemGenerationByAvailableItemsDataset
- InteractionsWithPopularityBasedNegativeSamplingDataset
- InteractionsWithHardNegativeMiningDataset
- InteractionsWithInBatchNegativesDataset
- InteractionsWithTimeAwareNegativeSamplingDataset
- InteractionsWithCurriculumNegativeSamplingDataset
- InteractionsWithDynamicNegativeMiningCandidateProviderDataset

Every sampler shares the same contract, checked for each one: ``__len__``
stretches the epoch by ``negative_proportion``, indices ``< n`` return
the positives unchanged, and the rest are negatives labelled ``0``.
"""

from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mars_rec.data.dataset import (
    InteractionsWithCurriculumNegativeSamplingDataset,
    InteractionsWithDynamicNegativeMiningCandidateProviderDataset,
    InteractionsWithHardNegativeMiningDataset,
    InteractionsWithInBatchNegativesDataset,
    InteractionsWithNegativeItemGenerationByAvailableItemsDataset,
    InteractionsWithPopularityBasedNegativeSamplingDataset,
    InteractionsWithTimeAwareNegativeSamplingDataset,
)
from mars_rec.filtering import CandidateFilter, CompositeProvider
from mars_rec.meta_config import Column, IOType, ProjectConfig

ITEMS = [2, 3, 4, 5, 6, 7, 8]  # framework idx (0/1 reserved)


@pytest.fixture
def config():
    return ProjectConfig(
        base_dir="/tmp/test",
        prepare_data_frames_task=MagicMock(),
        dataset_class=MagicMock(),
        user_column=Column("user_id", IOType.INDEXABLE),
        item_column=Column("item_id", IOType.INDEXABLE),
        other_input_columns=[],
        output_column=Column("label", IOType.NUMBER),
    )


@pytest.fixture
def index_mapping():
    return {
        "user_id": defaultdict(int, {"1": 1, "2": 2, "3": 3}),
        "item_id": defaultdict(int, {str(10 * (i - 1)): i for i in ITEMS}),  # "10"→2 … "70"→8
    }


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 1, 2, 3, 1],
            "item_id": ITEMS,
            "label": [1.0] * 7,
        }
    )


def _split(ds):
    """Materialise the whole epoch and split positives / negatives."""
    (user, item), output = ds[list(range(len(ds)))]
    pos = output == 1.0
    return user, item, output, pos


def _assert_contract(ds, n_rows, n_neg):
    assert len(ds) == n_rows + n_neg
    user, item, output, pos = _split(ds)
    assert pos.sum() == n_rows
    assert (~pos).sum() == n_neg
    np.testing.assert_array_equal(item[:n_rows], ITEMS[:n_rows])
    assert item.dtype.kind == "i"
    return item[~pos]


def _assert_index_forms(ds, n):
    """int / slice / positive-only / negative-only indexing all work."""
    (_, item), out = ds[0]
    assert item.tolist() == [ITEMS[0]] and out.tolist() == [1.0]
    (_, item), out = ds[slice(0, 2)]
    assert item.tolist() == ITEMS[:2]
    (_, item), out = ds[[n, n + 1]]
    assert out.tolist() == [0.0, 0.0] and len(item) == 2


# ---------------------------------------------------------------------------
# ByAvailableItems (slate-restricted)
# ---------------------------------------------------------------------------


class TestByAvailableItems:
    @pytest.fixture
    def config(self, config):
        config.available_items_column_name = "available_items"
        return config

    @pytest.fixture
    def slate_df(self, df):
        out = df.copy()
        # Each row's slate: its own item plus "60"/"70" (idx 7, 8).
        out["available_items"] = [[str(10 * (i - 1)), "60", "70"] for i in ITEMS]
        return out

    def test_negatives_come_from_row_slate(self, config, index_mapping, slate_df):
        ds = InteractionsWithNegativeItemGenerationByAvailableItemsDataset(
            slate_df, None, config, index_mapping, negative_proportion=0.5
        )
        neg = _assert_contract(ds, 7, 7)
        # Negative k is built from source row k % n and must avoid its positive.
        for k, it in enumerate(neg.tolist()):
            assert it in {7, 8} and it != ITEMS[k]
        _assert_index_forms(ds, 7)

    def test_requires_slate_column(self, config, index_mapping, df):
        with pytest.raises(AssertionError):
            InteractionsWithNegativeItemGenerationByAvailableItemsDataset(df, None, config, index_mapping)


# ---------------------------------------------------------------------------
# Popularity-based
# ---------------------------------------------------------------------------


class TestPopularityBased:
    def test_contract_and_frequency_distribution(self, config, index_mapping):
        # Item 2 appears 3×, others once.
        df = pd.DataFrame({"user_id": [1, 2, 3, 1, 2], "item_id": [2, 2, 2, 3, 4], "label": [1.0] * 5})
        ds = InteractionsWithPopularityBasedNegativeSamplingDataset(
            df, None, config, index_mapping, negative_proportion=0.5, smoothing_factor=1.0
        )
        np.testing.assert_allclose(ds._item_probabilities, [0, 0, 0.6, 0.2, 0.2])
        np.random.seed(0)
        (_, item), out = ds[list(range(len(ds)))]
        neg = item[out == 0.0]
        assert len(neg) == 5
        assert set(neg.tolist()) <= {2, 3, 4}  # never a reserved / zero-prob slot

    def test_external_popularity_column(self, config, index_mapping, df):
        df = df.assign(pop=[0.0, 0.0, 0.0, 0.0, 0.0, -5.0, 9.0])  # negatives clipped to 0
        ds = InteractionsWithPopularityBasedNegativeSamplingDataset(
            df, None, config, index_mapping, smoothing_factor=1.0, popularity_column="pop"
        )
        expected = np.zeros(9)
        expected[8] = 1.0
        np.testing.assert_allclose(ds._item_probabilities, expected)

    def test_all_zero_popularity_falls_back_to_uniform(self, config, index_mapping, df):
        ds = InteractionsWithPopularityBasedNegativeSamplingDataset(
            df.assign(pop=0.0), None, config, index_mapping, popularity_column="pop"
        )
        # Uniform over the real items 2..8 only; slots 0/1 get no mass.
        expected = np.full(9, 1 / 7)
        expected[:2] = 0.0
        np.testing.assert_allclose(ds._item_probabilities, expected)

    def test_negatives_differ_from_source(self, config, index_mapping, df):
        ds = InteractionsWithPopularityBasedNegativeSamplingDataset(
            df, None, config, index_mapping, negative_proportion=0.5
        )
        np.random.seed(1)
        _assert_contract(ds, 7, 7)
        _assert_index_forms(ds, 7)


# ---------------------------------------------------------------------------
# Hard negative mining (pre-computed column)
# ---------------------------------------------------------------------------


class TestHardNegativeMining:
    def test_all_hard_negatives_come_from_column(self, config, index_mapping, df):
        # Stored as strings, like a CSV round-trip; row 0 has none.
        df = df.assign(hard_negatives=["[]"] + ["[7, 8]"] * 6)
        ds = InteractionsWithHardNegativeMiningDataset(
            df, None, config, index_mapping, negative_proportion=0.5, hard_negative_ratio=1.0
        )
        assert isinstance(ds._hard_negatives[1], list)
        np.random.seed(0)
        neg = _assert_contract(ds, 7, 7)
        assert set(neg.tolist()) <= set(ITEMS)
        _assert_index_forms(ds, 7)

    def test_hard_candidates_used_when_present(self, config, index_mapping, df):
        df = df.assign(hard_negatives=[[7, 8]] * 7)
        ds = InteractionsWithHardNegativeMiningDataset(
            df, None, config, index_mapping, negative_proportion=0.5, hard_negative_ratio=1.0
        )
        np.random.seed(0)
        (_, item), out = ds[list(range(len(ds)))]
        assert set(item[out == 0.0].tolist()) <= {7, 8}

    def test_random_fallback_without_column(self, config, index_mapping, df):
        ds = InteractionsWithHardNegativeMiningDataset(df, None, config, index_mapping, negative_proportion=0.5)
        assert ds._has_hard_negatives is False
        np.random.seed(0)
        neg = _assert_contract(ds, 7, 7)
        assert set(neg.tolist()) <= set(ITEMS)


# ---------------------------------------------------------------------------
# In-batch negatives
# ---------------------------------------------------------------------------


class TestInBatchNegatives:
    def test_positives_only_expose_batch_items(self, config, index_mapping, df):
        ds = InteractionsWithInBatchNegativesDataset(df, None, config, index_mapping)
        assert len(ds) == 7
        (_, item), (out, in_batch) = ds[slice(0, 3)]
        np.testing.assert_array_equal(item, ITEMS[:3])
        np.testing.assert_array_equal(in_batch, ITEMS[:3])
        np.testing.assert_array_equal(out, [1.0, 1.0, 1.0])
        (_, item), (out, in_batch) = ds[4]
        assert item.tolist() == [ITEMS[4]] and in_batch.tolist() == [ITEMS[4]]

    def test_with_random_negatives(self, config, index_mapping, df):
        ds = InteractionsWithInBatchNegativesDataset(
            df, None, config, index_mapping, include_random_negatives=True, random_negative_proportion=0.5
        )
        assert len(ds) == 14
        np.random.seed(0)
        (_, item), (out, in_batch) = ds[list(range(14))]
        assert (out == 0.0).sum() == 7
        np.testing.assert_array_equal(in_batch, ITEMS)  # only the positives
        (_, item), (out, in_batch) = ds[[7, 8]]
        assert out.tolist() == [0.0, 0.0] and in_batch.size == 0

    def test_item_not_input_returns_empty_batch_items(self, config, index_mapping, df):
        config.item_is_input = False
        ds = InteractionsWithInBatchNegativesDataset(df, None, config, index_mapping)
        _, (_, in_batch) = ds[[0, 1]]
        assert in_batch.size == 0


# ---------------------------------------------------------------------------
# Time-aware
# ---------------------------------------------------------------------------


class TestTimeAware:
    @pytest.fixture
    def ts_df(self, df):
        # Item ITEMS[k] first appears at t = k.
        return df.assign(timestamp=list(range(7)))

    def test_negatives_existed_at_interaction_time(self, config, index_mapping, ts_df):
        ds = InteractionsWithTimeAwareNegativeSamplingDataset(
            ts_df, None, config, index_mapping, negative_proportion=0.5
        )
        np.random.seed(0)
        _assert_contract(ds, 7, 7)
        _assert_index_forms(ds, 7)
        for _ in range(50):
            # At t=3 only items 2..5 exist.
            assert ds._sample_item_before(3, exclude=5) in {2, 3, 4}

    def test_single_or_no_eligible_item(self, config, index_mapping, ts_df):
        ds = InteractionsWithTimeAwareNegativeSamplingDataset(ts_df, None, config, index_mapping)
        assert ds._sample_item_before(0, exclude=9) == 2  # only item 2 exists at t=0
        # Before any item exists → uniform fallback over the catalog.
        assert ds._sample_item_before(-1, exclude=2) in set(ITEMS) - {2}

    def test_rejection_exhausted_returns_other_item(self, config, index_mapping, ts_df, monkeypatch):
        ds = InteractionsWithTimeAwareNegativeSamplingDataset(ts_df, None, config, index_mapping)
        # Force every random draw to land on the excluded item (position 0).
        monkeypatch.setattr(np.random, "randint", lambda lo, hi: 0)
        assert ds._sample_item_before(1, exclude=2) == 3
        assert ds._sample_item_before(1, exclude=4) == 2

    def test_explicit_first_seen_mapping(self, config, index_mapping, ts_df):
        ds = InteractionsWithTimeAwareNegativeSamplingDataset(
            ts_df, None, config, index_mapping, item_first_seen={8: 0.0, 99: 0.0}
        )
        assert ds._sample_item_before(0.5, exclude=3) == 8

    def test_without_timestamps_uses_uniform(self, config, index_mapping, df):
        ds = InteractionsWithTimeAwareNegativeSamplingDataset(df, None, config, index_mapping, negative_proportion=0.5)
        assert ds._timestamps is None and ds._item_first_seen is None
        assert ds._sample_item_before(0, exclude=2) in set(ITEMS) - {2}
        np.random.seed(0)
        _assert_contract(ds, 7, 7)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


class TestCurriculum:
    def test_difficulty_schedule(self, config, index_mapping, df):
        ds = InteractionsWithCurriculumNegativeSamplingDataset(
            df, None, config, index_mapping, total_epochs=5, warmup_epochs=1, smoothing_factor=1.0
        )
        np.testing.assert_allclose(ds._current_probs, ds._uniform_probs)
        ds.set_epoch(0)
        assert ds._difficulty == 0.0
        ds.set_epoch(3)
        assert ds._difficulty == pytest.approx(0.5)
        ds.set_epoch(10)
        assert ds._difficulty == 1.0
        np.testing.assert_allclose(ds._current_probs, ds._popularity_probs)
        assert ds._current_probs.sum() == pytest.approx(1.0)

    def test_full_difficulty_never_samples_unseen_items(self, config, index_mapping, df):
        ds = InteractionsWithCurriculumNegativeSamplingDataset(
            df, None, config, index_mapping, negative_proportion=0.5, total_epochs=2, warmup_epochs=0
        )
        ds.set_epoch(5)
        np.random.seed(0)
        neg = _assert_contract(ds, 7, 7)
        assert set(neg.tolist()) <= set(ITEMS)
        _assert_index_forms(ds, 7)


# ---------------------------------------------------------------------------
# DNS over a CandidateProvider pool
# ---------------------------------------------------------------------------


class _OddOnly(CandidateFilter):
    def mask(self, row, full_catalog):
        return (full_catalog % 2) == 1


class _PerUserFilter(CandidateFilter):
    """User 1 → only item 3; user 2 → nothing; others → odd items."""

    def mask(self, row, full_catalog):
        user = row.get("user_id")
        if user == 1:
            return full_catalog == 3
        if user == 2:
            return np.zeros(len(full_catalog), dtype=bool)
        return (full_catalog % 2) == 1


def _provider_config(config, flt):
    config.candidate_provider_factory = lambda pc, im, extras: CompositeProvider(filters=[flt])
    return config


class TestDNSCandidateProvider:
    def test_requires_provider(self, config, index_mapping, df):
        config.available_items_column_name = None
        with pytest.raises(ValueError, match="candidate_provider_factory"):
            InteractionsWithDynamicNegativeMiningCandidateProviderDataset(df, None, config, index_mapping)

    def test_precompute_caches_pool_and_drops_provider(self, config, index_mapping, df):
        ds = InteractionsWithDynamicNegativeMiningCandidateProviderDataset(
            df, None, _provider_config(config, _OddOnly()), index_mapping, negative_proportion=0.5
        )
        assert ds._provider is None and ds._df_for_rows is None and ds._full_catalog is None
        assert len(ds._row_eligibles) == 7
        assert all(e.tolist() == [3, 5, 7] for e in ds._row_eligibles)
        np.random.seed(0)
        (_, item), out = ds[list(range(len(ds)))]
        assert set(item[out == 0.0].tolist()) <= {3, 5, 7}

    def test_eligibles_cap(self, config, index_mapping, df):
        ds = InteractionsWithDynamicNegativeMiningCandidateProviderDataset(
            df, None, _provider_config(config, _OddOnly()), index_mapping, eligibles_cap=2
        )
        assert all(e.size == 2 and set(e.tolist()) <= {3, 5, 7} for e in ds._row_eligibles)

    @pytest.mark.parametrize("precompute", [True, False])
    def test_draw_candidates_pool_sizes(self, config, index_mapping, df, precompute):
        ds = InteractionsWithDynamicNegativeMiningCandidateProviderDataset(
            df,
            None,
            _provider_config(config, _PerUserFilter()),
            index_mapping,
            precompute_eligibles=precompute,
        )
        assert (ds._provider is None) is precompute
        np.random.seed(0)
        # Rows 0/1/2 belong to users 1/2/3.
        cands = ds._draw_candidates([0, 0, 1, 2, 2], np.array([2, 3, 2, 2, 3]), K=4)
        assert cands.shape == (5, 4)
        assert cands[0].tolist() == [3, 3, 3, 3]  # pool of one → with replacement
        assert all(c in ITEMS and c != 3 for c in cands[1])  # pool empty after exclude → uniform
        assert all(c in ITEMS and c != 2 for c in cands[2])  # empty pool → uniform
        assert set(cands[3].tolist()) <= {3, 5, 7}
        assert set(cands[4].tolist()) <= {5, 7}  # 3 is excluded

    def test_draw_candidates_without_replacement_when_pool_is_large(self, config, index_mapping, df):
        ds = InteractionsWithDynamicNegativeMiningCandidateProviderDataset(
            df, None, _provider_config(config, _OddOnly()), index_mapping
        )
        cands = ds._draw_candidates([3], np.array([2]), K=3)
        assert sorted(cands[0].tolist()) == [3, 5, 7]

    def test_legacy_lookup_tolerates_out_of_range_row(self, config, index_mapping, df):
        ds = InteractionsWithDynamicNegativeMiningCandidateProviderDataset(
            df, None, _provider_config(config, _OddOnly()), index_mapping, precompute_eligibles=False
        )
        # Row 99 does not exist → empty row dict, provider still returns the odd pool.
        cands = ds._draw_candidates([99], np.array([2]), K=2)
        assert set(cands[0].tolist()) <= {3, 5, 7}
