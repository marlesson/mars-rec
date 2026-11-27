"""Batch negative samplers must match the scalar ones in distribution and edge cases."""

import numpy as np
import pytest

from mars_rec.data.dataset import (
    _choose_from_pool_except_batch,
    _choose_row_except_batch,
    _pack_ragged,
)


class TestChooseFromPoolExceptBatch:
    def test_never_returns_the_exception(self):
        np.random.seed(0)
        pool = np.array([3, 4, 5], dtype=np.int64)
        exceptions = np.random.choice(pool, size=5000)
        picks = _choose_from_pool_except_batch(pool, exceptions)
        assert picks.dtype == np.int64
        assert not (picks == exceptions).any()
        assert set(np.unique(picks)) <= set(pool.tolist())

    def test_uniform_over_the_rest_of_the_pool(self):
        np.random.seed(1)
        pool = np.arange(3, 8, dtype=np.int64)  # 5 items
        picks = _choose_from_pool_except_batch(pool, np.full(40_000, 5, dtype=np.int64))
        freqs = np.array([(picks == v).mean() for v in (3, 4, 6, 7)])
        assert (picks != 5).all()
        np.testing.assert_allclose(freqs, 0.25, atol=0.02)

    def test_degenerate_pools(self):
        excl = np.array([7, 8], dtype=np.int64)
        assert _choose_from_pool_except_batch(np.array([], dtype=np.int64), excl).tolist() == [7, 8]
        assert _choose_from_pool_except_batch(np.array([9], dtype=np.int64), excl).tolist() == [9, 9]
        # Two-item pool where every row excludes one of them: always the other.
        assert _choose_from_pool_except_batch(np.array([3, 4]), np.array([3, 4, 3])).tolist() == [4, 3, 4]


class TestChooseRowExceptBatch:
    def test_ragged_rows_respect_length_and_exception(self):
        np.random.seed(2)
        rows = [[3, 4, 5], [6], [], [7, 7, 8], [9, 10]]
        matrix, lengths = _pack_ragged(rows)
        assert lengths.tolist() == [3, 1, 0, 3, 2]

        idx = np.repeat(np.arange(len(rows)), 2000)
        exceptions = np.array([3, 6, 11, 8, 9])[idx]
        picks = _choose_row_except_batch(matrix, lengths, idx, exceptions)

        for r, values in enumerate(rows):
            got = picks[idx == r]
            if r == 1:  # only the exception is available -> first value, like _choose_except
                assert set(got.tolist()) == {6}
            elif r == 2:  # empty row -> the exception itself
                assert set(got.tolist()) == {11}
            else:
                assert set(got.tolist()) <= set(values) - {exceptions[idx == r][0]}

    def test_duplicates_keep_their_weight(self):
        np.random.seed(3)
        matrix, lengths = _pack_ragged([[7, 7, 7, 8, 9]])
        picks = _choose_row_except_batch(matrix, lengths, np.zeros(30_000, dtype=np.int64), np.full(30_000, 9))
        assert (picks != 9).all()
        assert (picks == 7).mean() == pytest.approx(0.75, abs=0.02)

    def test_pack_rejects_non_integer_rows(self):
        assert _pack_ragged([["a", "b"]]) is None
