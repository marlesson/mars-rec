"""Tests for ``mars_rec.torch.data``.

Custom DataLoader plumbing that the framework relies on for performance:
- ``FasterBatchSampler`` yields lists of indices, supports shuffle and
  weighted sampling.
- ``NoAutoCollationDataLoader`` disables PyTorch's auto-collation so
  custom datasets that return pre-packed numpy arrays (the InteractionsDataset
  pattern) don't get re-converted per batch.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from mars_rec.torch.data import FasterBatchSampler, NoAutoCollationDataLoader


class _SimpleDataset(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Mimic InteractionsDataset: idx may be a list → return whole batch.
        if isinstance(idx, (list, np.ndarray)):
            arr = np.asarray(idx, dtype=np.int64)
            return (arr,), arr.astype(np.float32)
        return (np.array([idx], dtype=np.int64),), np.float32(idx)


# ---------------------------------------------------------------------------
# FasterBatchSampler
# ---------------------------------------------------------------------------

class TestFasterBatchSampler:
    def test_yields_expected_batch_sizes_no_drop_last(self):
        ds = _SimpleDataset(25)
        sampler = FasterBatchSampler(ds, batch_size=8, drop_last=False)
        batches = list(sampler)
        assert [len(b) for b in batches] == [8, 8, 8, 1]
        # All indices unique and cover the full range.
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(25))

    def test_drop_last_drops_partial_batch(self):
        ds = _SimpleDataset(25)
        sampler = FasterBatchSampler(ds, batch_size=8, drop_last=True)
        batches = list(sampler)
        assert [len(b) for b in batches] == [8, 8, 8]

    def test_len_matches_iteration(self):
        ds = _SimpleDataset(20)
        sampler = FasterBatchSampler(ds, batch_size=7, drop_last=False)
        assert len(sampler) == 3  # ceil(20/7)
        assert len(list(sampler)) == 3

    def test_shuffle_reproducible_with_seed(self):
        ds = _SimpleDataset(50)
        torch.manual_seed(42)
        batches_a = list(FasterBatchSampler(ds, batch_size=10, shuffle=True))
        torch.manual_seed(42)
        batches_b = list(FasterBatchSampler(ds, batch_size=10, shuffle=True))
        assert batches_a == batches_b

    def test_shuffle_off_returns_indices_in_order(self):
        ds = _SimpleDataset(10)
        sampler = FasterBatchSampler(ds, batch_size=4, shuffle=False)
        assert list(sampler) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]

    def test_sample_weights_must_match_dataset_length(self):
        ds = _SimpleDataset(10)
        with pytest.raises(ValueError, match="length"):
            FasterBatchSampler(ds, batch_size=4, sample_weights=torch.ones(5))

    def test_sample_weights_must_be_non_negative(self):
        ds = _SimpleDataset(5)
        with pytest.raises(ValueError, match="non-negative"):
            FasterBatchSampler(
                ds, batch_size=2, sample_weights=torch.tensor([1.0, -0.1, 1.0, 1.0, 1.0])
            )

    def test_sample_weights_zero_sum_raises(self):
        ds = _SimpleDataset(5)
        with pytest.raises(ValueError, match="positive value"):
            FasterBatchSampler(
                ds, batch_size=2, sample_weights=torch.zeros(5)
            )

    def test_sample_weights_biases_sampling(self):
        """A weight vector dominated by one index → that index dominates samples."""
        torch.manual_seed(7)
        ds = _SimpleDataset(10)
        w = torch.zeros(10)
        w[3] = 1.0  # only idx 3 has weight
        sampler = FasterBatchSampler(ds, batch_size=5, sample_weights=w)
        flat = [i for b in sampler for i in b]
        # multinomial with replacement → all sampled indices must equal 3.
        assert flat == [3] * len(flat)


# ---------------------------------------------------------------------------
# NoAutoCollationDataLoader
# ---------------------------------------------------------------------------

class TestNoAutoCollationDataLoader:
    def test_auto_collation_disabled(self):
        ds = _SimpleDataset(10)
        sampler = FasterBatchSampler(ds, batch_size=4, shuffle=False)
        loader = NoAutoCollationDataLoader(ds, batch_sampler=sampler, num_workers=0)
        # The whole point: PyTorch's default collation is bypassed.
        assert loader._auto_collation is False

    def test_index_sampler_returns_batch_sampler(self):
        ds = _SimpleDataset(10)
        sampler = FasterBatchSampler(ds, batch_size=4, shuffle=False)
        loader = NoAutoCollationDataLoader(ds, batch_sampler=sampler, num_workers=0)
        assert loader._index_sampler is sampler

    def test_getitem_called_once_per_batch_not_per_sample(self):
        """The point of disabling auto-collation: __getitem__ receives the
        whole batch (a list of indices) in one call, not B separate calls.
        InteractionsDataset uses this to pre-pack numpy arrays once.
        """
        calls = []

        class _CountingDS(Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                calls.append(idx)
                arr = np.asarray(idx, dtype=np.int64)
                return (arr,), arr.astype(np.float32)

        ds = _CountingDS()
        sampler = FasterBatchSampler(ds, batch_size=4, shuffle=False)
        loader = NoAutoCollationDataLoader(ds, batch_sampler=sampler, num_workers=0)
        list(loader)
        # 8 rows / batch_size 4 = 2 calls (not 8)
        assert len(calls) == 2
        assert calls[0] == [0, 1, 2, 3]
        assert calls[1] == [4, 5, 6, 7]
