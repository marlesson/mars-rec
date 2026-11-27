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
            filters=[_AllFalse()], min_pool=1, fallback=custom,
        )
        out = p.get_candidates({"backup": [99]}, CATALOG)
        np.testing.assert_array_equal(out, np.array([99], dtype=np.int64))

    def test_name_includes_filter_names(self):
        p = CompositeProvider(
            filters=[_EvenIdxFilter(), _ThresholdFilter()], min_pool=0,
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
