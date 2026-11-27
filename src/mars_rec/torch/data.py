"""Custom PyTorch ``Sampler`` and ``DataLoader`` used by the training loop.

:class:`FasterBatchSampler` yields lists of indices instead of one
index at a time, which lets :class:`NoAutoCollationDataLoader` skip
PyTorch's default per-sample collation step entirely — the wrapped
:class:`InteractionsDataset` already returns whole-batch numpy arrays,
so collation would otherwise re-do work the Dataset has already done.
"""

from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Sampler, Dataset


class FasterBatchSampler(Sampler):
    """Batch sampler that yields ``list[int]`` chunks of size ``batch_size``."""

    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = False,
        sample_weights: Optional[torch.Tensor] = None,
    ):
        """Batched sampler that returns lists of indices.

        Set ``shuffle=True`` for a uniform random permutation. Set
        ``sample_weights`` to a 1-D tensor of length ``len(data_source)``
        to draw indices via ``torch.multinomial`` (with replacement) —
        useful for re-weighting the training distribution when one
        sub-cohort dominates and we want the model to see another
        cohort more often (e.g. matching a leaderboard regime).
        ``sample_weights`` overrides ``shuffle``.
        """
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        if sample_weights is not None:
            sample_weights = torch.as_tensor(sample_weights, dtype=torch.float)
            if sample_weights.ndim != 1 or len(sample_weights) != len(data_source):
                raise ValueError(
                    f"sample_weights must be a 1-D tensor of length "
                    f"{len(data_source)}, got shape={tuple(sample_weights.shape)}"
                )
            if (sample_weights < 0).any():
                raise ValueError("sample_weights must be non-negative")
            if float(sample_weights.sum()) <= 0:
                raise ValueError("sample_weights must sum to a positive value")
        self.sample_weights = sample_weights

    @property
    def num_samples(self):
        if not hasattr(self, "_num_samples"):
            self._num_samples = len(self.data_source)
        return self._num_samples

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.sample_weights is not None:
            # Weighted sampling with replacement — same row count per epoch
            # so the trainer sees a consistent number of batches.
            iter_list: List[int] = torch.multinomial(
                self.sample_weights,
                self.num_samples,
                replacement=True,
            ).tolist()
        elif self.shuffle:
            iter_list = torch.randperm(self.num_samples).tolist()
        else:
            iter_list = list(range(self.num_samples))
        for i in range(0, self.num_samples, self.batch_size):
            last_idx = i + self.batch_size
            if last_idx < self.num_samples or not self.drop_last:
                yield iter_list[i:last_idx]


class NoAutoCollationDataLoader(DataLoader):
    """``DataLoader`` that disables PyTorch's auto-collation step.

    Pair with :class:`FasterBatchSampler`: the Dataset's
    ``__getitem__`` already returns whole-batch numpy arrays, so the
    default per-sample collation would just re-stack what's already
    stacked. Skipping it is a measurable speedup on wide
    schemas (~2-3× per-batch on a 37-column config).

    When ``num_workers > 0``, defaults ``persistent_workers=True`` and
    ``prefetch_factor=4`` so per-worker initialisation (HuggingFace
    metadata, candidate-feature lookups) does not repeat every epoch.
    """

    def __init__(self, *args, **kwargs):
        # Default to keeping workers alive across epochs and prefetching
        # extra batches — only meaningful when num_workers > 0. Without
        # this, every epoch boundary tears down all workers and rebuilds
        # them, which forces re-initialisation of any per-worker state
        # (HuggingFace metadata loads, candidate-feature lookups, etc).
        if kwargs.get("num_workers", 0) > 0:
            kwargs.setdefault("persistent_workers", True)
            kwargs.setdefault("prefetch_factor", 4)
        super().__init__(*args, **kwargs)

    @property
    def _auto_collation(self):
        return False

    @property
    def _index_sampler(self):
        return self.batch_sampler
