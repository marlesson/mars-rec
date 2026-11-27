"""Unit tests for ``mars_rec.filtering``."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from mars_rec.filtering import (
    AvailableItemsColumnProvider,
    CandidateFilter,
    CandidateProvider,
    CompositeProvider,
    FullCatalogProvider,
    resolve_candidate_provider,
)


CATALOG = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)


# ---------------------------------------------------------------------------
# FullCatalogProvider
# ---------------------------------------------------------------------------


class TestFullCatalogProvider:
    def test_returns_input_unchanged(self):
        p = FullCatalogProvider()
        out = p.get_candidates({"any": "row"}, CATALOG)
        assert out is CATALOG  # no-op should preserve identity

    def test_name(self):
        assert FullCatalogProvider().name() == "FullCatalogProvider"


# ---------------------------------------------------------------------------
# AvailableItemsColumnProvider
# ---------------------------------------------------------------------------


class TestAvailableItemsColumnProvider:
    def test_reads_column(self):
        p = AvailableItemsColumnProvider("avail")
        row = {"avail": [3, 5, 7]}
        out = p.get_candidates(row, CATALOG)
        np.testing.assert_array_equal(out, np.array([3, 5, 7], dtype=np.int64))
        assert out.dtype == np.int64

    def test_missing_column_returns_full_catalog(self):
        p = AvailableItemsColumnProvider("avail")
        out = p.get_candidates({}, CATALOG)
        assert out is CATALOG

    def test_empty_column_returns_full_catalog(self):
        p = AvailableItemsColumnProvider("avail")
        out = p.get_candidates({"avail": []}, CATALOG)
        assert out is CATALOG


# ---------------------------------------------------------------------------
# CandidateFilter stubs for CompositeProvider
# ---------------------------------------------------------------------------


class _EvenIdxFilter(CandidateFilter):
    """Keep even idx values."""

    def mask(self, row, full_catalog):
        return (full_catalog % 2) == 0


class _ThresholdFilter(CandidateFilter):
    """Keep idx >= threshold (read from row)."""

    def mask(self, row, full_catalog):
        thr = row.get("thr", 0)
        return full_catalog >= thr


class _AllFalse(CandidateFilter):
    """Reject everything (forces fallback)."""

    def mask(self, row, full_catalog):
        return np.zeros(len(full_catalog), dtype=bool)


class TestCompositeProvider:
    def test_no_filters_is_noop(self):
        p = CompositeProvider(filters=[])
        out = p.get_candidates({}, CATALOG)
        assert out is CATALOG

    def test_chains_with_and(self):
        p = CompositeProvider(filters=[_EvenIdxFilter(), _ThresholdFilter()])
        out = p.get_candidates({"thr": 5}, CATALOG)
        # even & >=5  → 6, 8
        np.testing.assert_array_equal(out, np.array([6, 8], dtype=np.int64))

    def test_min_pool_triggers_fallback(self):
        # filter survivors: even & >=5 → {6, 8}. min_pool=3 forces fallback.
        p = CompositeProvider(
            filters=[_EvenIdxFilter(), _ThresholdFilter()],
            min_pool=3,
        )
        out = p.get_candidates({"thr": 5}, CATALOG)
        assert out is CATALOG

    def test_all_false_falls_back(self):
        p = CompositeProvider(filters=[_AllFalse()], min_pool=1)
        out = p.get_candidates({}, CATALOG)
        assert out is CATALOG

    def test_custom_fallback(self):
        custom = AvailableItemsColumnProvider("backup")
        p = CompositeProvider(
            filters=[_AllFalse()],
            min_pool=1,
            fallback=custom,
        )
        out = p.get_candidates({"backup": [99]}, CATALOG)
        np.testing.assert_array_equal(out, np.array([99], dtype=np.int64))

    def test_name_includes_filter_names(self):
        p = CompositeProvider(
            filters=[_EvenIdxFilter(), _ThresholdFilter()],
            min_pool=0,
        )
        assert "_EvenIdxFilter" in p.name()
        assert "_ThresholdFilter" in p.name()


# ---------------------------------------------------------------------------
# resolve_candidate_provider — wiring through ProjectConfig
# ---------------------------------------------------------------------------


class _FakeProjectConfig:
    """Minimal stand-in exposing the two attributes resolve consults."""

    def __init__(self, factory=None, available_items_column_name=None):
        self.candidate_provider_factory = factory
        self.available_items_column_name = available_items_column_name


class TestResolveCandidateProvider:
    def test_factory_takes_precedence(self):
        sentinel = FullCatalogProvider()

        def fac(pc, idx, extras):
            return sentinel

        pc = _FakeProjectConfig(factory=fac, available_items_column_name="avail")
        out = resolve_candidate_provider(pc, {}, {})
        assert out is sentinel

    def test_legacy_column_fallback(self):
        pc = _FakeProjectConfig(factory=None, available_items_column_name="avail")
        out = resolve_candidate_provider(pc, {}, {})
        assert isinstance(out, AvailableItemsColumnProvider)
        assert out.column_name == "avail"

    def test_returns_none_when_neither_configured(self):
        pc = _FakeProjectConfig(factory=None, available_items_column_name=None)
        out = resolve_candidate_provider(pc, {}, {})
        assert out is None

    def test_factory_receives_index_mapping_and_extras(self):
        captured = {}

        def fac(pc, idx, extras):
            captured["idx"] = idx
            captured["extras"] = extras
            return FullCatalogProvider()

        pc = _FakeProjectConfig(factory=fac)
        idx_map = {"item": {"a": 0}}
        extras = {"foo": "bar"}
        resolve_candidate_provider(pc, idx_map, extras)
        assert captured["idx"] is idx_map
        assert captured["extras"] is extras


# ---------------------------------------------------------------------------
# Union filters, ranking and score fusion
# ---------------------------------------------------------------------------


class _ScoreFilter(CandidateFilter):
    """Mask-all filter that scores items with a fixed per-catalog array."""

    def __init__(self, scores, keep=None):
        self._scores = None if scores is None else np.asarray(scores, dtype=np.float64)
        self._keep = keep

    def mask(self, row, full_catalog):
        if self._keep is None:
            return np.ones(len(full_catalog), dtype=bool)
        return np.isin(full_catalog, self._keep)

    def score(self, row, full_catalog):
        return self._scores


class _EmptyProvider(CandidateProvider):
    def get_candidates(self, row, full_catalog):
        return np.asarray([], dtype=np.int64)


class TestCandidateFilterDefaults:
    def test_default_score_is_none(self):
        assert _EvenIdxFilter().score({}, CATALOG) is None

    def test_default_name(self):
        assert _EvenIdxFilter().name() == "_EvenIdxFilter"


class TestCompositeUnionFilters:
    def test_union_adds_to_intersection(self):
        # (even & >=7) | {3}  → 3, 8
        p = CompositeProvider(
            filters=[_EvenIdxFilter(), _ThresholdFilter()],
            union_filters=[_ScoreFilter(None, keep=[3])],
        )
        out = p.get_candidates({"thr": 7}, CATALOG)
        np.testing.assert_array_equal(out, np.array([3, 8], dtype=np.int64))

    def test_union_survives_empty_intersection(self):
        # The AND part rejects everything, but the union still contributes.
        p = CompositeProvider(
            filters=[_AllFalse(), _EvenIdxFilter()],
            union_filters=[_ScoreFilter(None, keep=[4, 9])],
        )
        out = p.get_candidates({}, CATALOG)
        np.testing.assert_array_equal(out, np.array([4, 9], dtype=np.int64))

    def test_union_only_starts_from_empty(self):
        p = CompositeProvider(filters=[], union_filters=[_ScoreFilter(None, keep=[5])])
        out = p.get_candidates({}, CATALOG)
        np.testing.assert_array_equal(out, np.array([5], dtype=np.int64))

    def test_name_mentions_union_filters(self):
        p = CompositeProvider(filters=[], union_filters=[_EvenIdxFilter()])
        assert p.name() == "Composite(+(_EvenIdxFilter), min_pool=0)"

    def test_empty_name(self):
        assert CompositeProvider(filters=[]).name() == "Composite(<empty>, min_pool=0)"


class TestFullCatalogRank:
    def test_non_positive_top_k_or_empty_catalog(self):
        p = FullCatalogProvider()
        assert p.rank({}, CATALOG, top_k=0).size == 0
        assert p.rank({}, np.asarray([], dtype=np.int64), top_k=3).size == 0

    def test_catalog_order_without_popularity(self):
        out = FullCatalogProvider().rank({}, CATALOG, top_k=3)
        np.testing.assert_array_equal(out, np.array([2, 3, 4], dtype=np.int64))
        assert out.dtype == np.int64

    def test_popularity_ordering(self):
        pop = {5: 10.0, 8: 30.0, 3: 20.0}
        out = FullCatalogProvider().rank({}, CATALOG, top_k=3, popularity_fallback=pop)
        np.testing.assert_array_equal(out, np.array([8, 3, 5], dtype=np.int64))

    def test_popularity_top_k_larger_than_catalog(self):
        small = np.array([2, 3], dtype=np.int64)
        out = FullCatalogProvider().rank({}, small, top_k=10, popularity_fallback={3: 1.0})
        np.testing.assert_array_equal(out, np.array([3, 2], dtype=np.int64))


class TestCompositeRank:
    # Scores aligned to CATALOG = [2, 3, 4, 5, 6, 7, 8, 9]
    SCORES = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6]

    def test_orders_by_single_filter_score(self):
        p = CompositeProvider(filters=[_ScoreFilter(self.SCORES)])
        out = p.rank({}, CATALOG, top_k=3)
        np.testing.assert_array_equal(out, np.array([3, 5, 7], dtype=np.int64))

    def test_scores_restricted_to_eligible_pool(self):
        # Even items only: 2(0.1) 4(0.3) 6(0.2) 8(0.4) → 8, 4, 6, 2
        p = CompositeProvider(filters=[_EvenIdxFilter(), _ScoreFilter(self.SCORES)])
        out = p.rank({}, CATALOG, top_k=10)
        np.testing.assert_array_equal(out, np.array([8, 4, 6, 2], dtype=np.int64))

    @pytest.mark.parametrize("fusion", ["rrf", "sum", "max"])
    def test_fusion_strategies_agree_when_sources_agree(self, fusion):
        doubled = [2 * s for s in self.SCORES]
        p = CompositeProvider(filters=[_ScoreFilter(self.SCORES), _ScoreFilter(doubled)])
        out = p.rank({}, CATALOG, top_k=2, fusion=fusion)
        np.testing.assert_array_equal(out, np.array([3, 5], dtype=np.int64))

    def test_unknown_fusion_raises(self):
        p = CompositeProvider(filters=[_ScoreFilter(self.SCORES)])
        with pytest.raises(ValueError, match="Unknown fusion"):
            p.rank({}, CATALOG, top_k=2, fusion="median")

    def test_misaligned_scores_are_ignored(self):
        # Wrong-length scores are skipped → falls through to catalog order.
        p = CompositeProvider(filters=[_ScoreFilter([1.0, 2.0])])
        out = p.rank({}, CATALOG, top_k=3)
        np.testing.assert_array_equal(out, np.array([2, 3, 4], dtype=np.int64))

    def test_popularity_fallback_without_scores(self):
        p = CompositeProvider(filters=[_EvenIdxFilter()])
        pop = {2: 1.0, 6: 5.0, 9: 100.0}  # 9 is not eligible
        out = p.rank({}, CATALOG, top_k=2, popularity_fallback=pop)
        np.testing.assert_array_equal(out, np.array([6, 2], dtype=np.int64))

    def test_catalog_order_without_any_signal(self):
        p = CompositeProvider(filters=[_EvenIdxFilter()])
        out = p.rank({}, CATALOG, top_k=10)
        np.testing.assert_array_equal(out, np.array([2, 4, 6, 8], dtype=np.int64))

    def test_min_pool_fallback_ranks_full_catalog(self):
        p = CompositeProvider(filters=[_AllFalse(), _ScoreFilter(self.SCORES)], min_pool=1)
        out = p.rank({}, CATALOG, top_k=1)
        np.testing.assert_array_equal(out, np.array([3], dtype=np.int64))

    def test_non_positive_top_k(self):
        p = CompositeProvider(filters=[_ScoreFilter(self.SCORES)])
        assert p.rank({}, CATALOG, top_k=0).size == 0

    def test_empty_pool_returns_empty(self):
        p = CompositeProvider(filters=[_AllFalse()], min_pool=1, fallback=_EmptyProvider())
        assert p.rank({}, CATALOG, top_k=3).size == 0

    def test_pool_outside_catalog_returns_empty(self):
        # A fallback can hand back indices absent from full_catalog.
        custom = AvailableItemsColumnProvider("backup")
        p = CompositeProvider(filters=[_AllFalse()], min_pool=1, fallback=custom)
        assert p.rank({"backup": [99]}, CATALOG, top_k=3).size == 0


class TestFuseScores:
    def test_empty_list(self):
        from mars_rec.filtering import _fuse_scores

        assert _fuse_scores([], fusion="rrf").size == 0

    def test_constant_source_normalizes_to_zero(self):
        from mars_rec.filtering import _fuse_scores

        out = _fuse_scores([np.array([3.0, 3.0, 3.0])], fusion="sum")
        np.testing.assert_array_equal(out, np.zeros(3))

    def test_max_takes_best_normalized_source(self):
        from mars_rec.filtering import _fuse_scores

        a = np.array([0.0, 1.0, 0.5])
        b = np.array([1.0, 0.0, 0.5])
        np.testing.assert_allclose(_fuse_scores([a, b], fusion="max"), [1.0, 1.0, 0.5])
        np.testing.assert_allclose(_fuse_scores([a, b], fusion="sum"), [1.0, 1.0, 1.0])
