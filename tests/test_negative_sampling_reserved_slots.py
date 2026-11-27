"""Regression tests for negative sampling and the reserved index-mapping slots.

- #13: negatives must never be the reserved slots of ``create_index_mapping``
  (``0`` unknown, ``1`` None/NaN, ``2`` the ``-1`` pad token).
- #14: the popularity-weighted samplers must not loop forever when the
  positive item holds all the probability mass.
"""

import threading
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mars_rec.data.dataset import (
    InteractionsWithCandidateProviderNegativeDataset,
    InteractionsWithCurriculumNegativeSamplingDataset,
    InteractionsWithDynamicNegativeMiningDataset,
    InteractionsWithHardNegativeMiningDataset,
    InteractionsWithInBatchNegativesDataset,
    InteractionsWithMixedHardNegativeDataset,
    InteractionsWithNegativeItemGenerationDataset,
    InteractionsWithPopularityBasedNegativeSamplingDataset,
    InteractionsWithTimeAwareNegativeSamplingDataset,
    _choice_except,
    _choose_from_pool_except,
)
from mars_rec.filtering import FullCatalogProvider
from mars_rec.meta_config import Column, IOType, ProjectConfig
from mars_rec.utils.index_mapping import create_index_mapping, is_reserved_key, real_indices

RESERVED = {0, 1, 2}


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
    # Real mapping: None/NaN → 1, -1/"-1" → 2, "a".."f" → 3..8.
    return {
        "user_id": create_index_mapping(["u1", "u2"]),
        "item_id": create_index_mapping(list("abcdef")),
    }


@pytest.fixture
def df(index_mapping):
    items = [index_mapping["item_id"][c] for c in "abcdef"]
    return pd.DataFrame({"user_id": [3, 4] * 3, "item_id": items, "label": [1.0] * 6})


def _negatives(ds, rounds=100):
    n = len(ds._data_frame)
    out = []
    for _ in range(rounds):
        (_, item), output = ds[list(range(n, len(ds)))]
        if isinstance(output, tuple):  # in-batch sampler returns (output, in_batch_items)
            output = output[0]
        assert (output == 0.0).all()
        out.append(item)
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestRealIndices:
    def test_sentinels_are_reserved(self):
        assert is_reserved_key(None) and is_reserved_key(float("nan"))
        assert is_reserved_key(-1) and is_reserved_key("-1")
        assert not is_reserved_key("a") and not is_reserved_key(0)

    def test_real_indices_skip_sentinels_and_unknown(self, index_mapping):
        item_map = index_mapping["item_id"]
        _ = item_map["never-seen"]  # defaultdict lookup inserts key → 0
        np.testing.assert_array_equal(real_indices(item_map), [3, 4, 5, 6, 7, 8])

    def test_choose_from_pool_except(self):
        pool = np.array([3, 4], dtype=np.int64)
        assert all(_choose_from_pool_except(pool, 3) == 4 for _ in range(20))
        assert _choose_from_pool_except(np.array([5]), 5) == 5
        assert _choose_from_pool_except(np.array([], dtype=np.int64), 7) == 7

    def test_choose_from_pool_except_exhausted_rejection(self, monkeypatch):
        draws = iter([0] * 32 + [0])
        monkeypatch.setattr(np.random, "randint", lambda lo, hi: next(draws))
        assert _choose_from_pool_except(np.array([3, 4]), 3) == 4

    def test_choice_except(self):
        assert _choice_except(np.array([0.0, 1.0]), 1) is None
        assert _choice_except(np.array([0.0, 0.5, 0.5]), 1) == 2
        assert _choice_except(np.array([0.0, 1.0]), 0) == 1  # exception has no mass


# ---------------------------------------------------------------------------
# #13 — reserved slots are never negatives
# ---------------------------------------------------------------------------


SAMPLERS = [
    (InteractionsWithNegativeItemGenerationDataset, {}),
    (InteractionsWithHardNegativeMiningDataset, {}),
    (InteractionsWithInBatchNegativesDataset, {"include_random_negatives": True, "random_negative_proportion": 0.5}),
    (InteractionsWithTimeAwareNegativeSamplingDataset, {}),
    (InteractionsWithCurriculumNegativeSamplingDataset, {}),  # epoch 0 → uniform
    (InteractionsWithPopularityBasedNegativeSamplingDataset, {"smoothing_factor": 0.0}),  # 0 ** 0 == 1
    (InteractionsWithDynamicNegativeMiningDataset, {"candidates_per_slot": 4}),
    (
        InteractionsWithMixedHardNegativeDataset,
        {"ann_ratio": 0.0, "popularity_ratio": 0.5, "uniform_ratio": 0.5, "smoothing_factor": 0.0},
    ),
]


@pytest.mark.parametrize("cls,kwargs", SAMPLERS, ids=[c.__name__ for c, _ in SAMPLERS])
def test_reserved_slots_never_sampled(cls, kwargs, config, index_mapping, df):
    if "random_negative_proportion" not in kwargs:
        kwargs = {**kwargs, "negative_proportion": 0.5}
    ds = cls(df, None, config, index_mapping, **kwargs)
    np.random.seed(0)
    neg = _negatives(ds)
    assert not set(neg.tolist()) & RESERVED
    assert set(neg.tolist()) <= {3, 4, 5, 6, 7, 8}


def test_provider_catalog_excludes_pad_slot(config, index_mapping, df):
    config.candidate_provider_factory = lambda pc, im, extras: FullCatalogProvider()
    ds = InteractionsWithCandidateProviderNegativeDataset(df, None, config, index_mapping, negative_proportion=0.5)
    np.testing.assert_array_equal(ds._full_catalog, [3, 4, 5, 6, 7, 8])
    np.random.seed(0)
    assert not set(_negatives(ds, rounds=20).tolist()) & RESERVED


def test_legacy_range_when_mapping_has_no_real_items(config, df):
    # A mapping without real entries keeps the old [0, max] behaviour.
    empty = {"user_id": {}, "item_id": {}}
    ds = InteractionsWithNegativeItemGenerationDataset(df, None, config, empty, negative_proportion=0.5)
    np.testing.assert_array_equal(ds._negative_item_pool, np.arange(0, 9))


# ---------------------------------------------------------------------------
# #14 — no infinite resampling loop
# ---------------------------------------------------------------------------


def _run_with_timeout(fn, seconds=5.0):
    result = {}

    def target():
        result["value"] = fn()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "sampler did not return (infinite resampling loop)"
    return result["value"]


def test_popularity_all_mass_on_positive_falls_back(config, index_mapping):
    # Item 3 holds all the popularity mass and is also the only positive.
    df = pd.DataFrame({"user_id": [3, 4], "item_id": [3, 4], "label": [1.0, 1.0], "pop": [5.0, 0.0]})
    ds = InteractionsWithPopularityBasedNegativeSamplingDataset(
        df, None, config, index_mapping, negative_proportion=0.5, smoothing_factor=1.0, popularity_column="pop"
    )
    np.testing.assert_allclose(ds._item_probabilities[3:5], [1.0, 0.0])
    np.random.seed(0)
    for _ in range(20):
        (_, item), _out = _run_with_timeout(lambda: ds[[2, 3]])
        # Sources are random rows; a source with positive 3 must fall back to 4.
        assert set(item.tolist()) <= {3, 4}


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (InteractionsWithPopularityBasedNegativeSamplingDataset, {}),
        (InteractionsWithCurriculumNegativeSamplingDataset, {"total_epochs": 1, "warmup_epochs": 0}),
    ],
)
def test_single_item_catalog_does_not_hang(cls, kwargs, config, index_mapping):
    df = pd.DataFrame({"user_id": [3, 3], "item_id": [3, 3], "label": [1.0, 1.0]})
    ds = cls(df, None, config, index_mapping, negative_proportion=0.5, **kwargs)
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(5)  # full difficulty → pure popularity
    (_, item), out = _run_with_timeout(lambda: ds[[2, 3]])
    assert out.tolist() == [0.0, 0.0]
    assert item.tolist() == [3, 3]  # degenerate catalog: nothing else to draw
