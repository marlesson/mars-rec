"""PyTorch Datasets wrapping the train / val / test CSVs.

This module hosts the ``Dataset`` classes a project plugs into
``ProjectConfig.dataset_class``. They share a common backbone
(:class:`InteractionsDataset`) and differ only in *how negatives are
generated* — uniform random, restricted to a per-row slate, popularity-
weighted, hard-mined via embeddings, in-batch, time-aware, curriculum,
or via a pluggable :class:`mars_rec.filtering.CandidateProvider`.

Pre-loading helpers
-------------------

- :func:`preprocess_interactions_data_frame` — runs once on the
  DataFrame before it reaches the Dataset constructor: stringifies the
  user / item columns and parses any array columns (FLOAT_ARRAY,
  INT_ARRAY, INDEXABLE_ARRAY, SEQUENCE) from their CSV-literal form.
- :func:`preprocess_metadata_data_frame` — converts the per-item
  metadata CSV into a ``{column_name: np.ndarray}`` lookup keyed by
  item index, consumed by :class:`InteractionsDataset` to attach the
  metadata vector at sample time.

Each ``__getitem__`` returns ``(inputs, output)`` where ``inputs`` is a
tuple of numpy arrays whose order matches
:attr:`ProjectConfig.input_columns`.
"""

from typing import Callable, Tuple, List, Union, Optional, Dict, Any

import contextlib
import functools
import inspect
import logging
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset

from mars_rec.data.candidate_features import (
    FN_PARAM,
    LOOKUP_PARAM,
    apply_candidate_features,
    convert_dtype,
    resolve_candidate_lookup,
    validate_candidate_feature_fn,
)
from mars_rec.meta_config import ProjectConfig, IOType, Column
from mars_rec.utils.index_mapping import map_array, real_indices
from mars_rec.utils.utils import parallel_literal_eval, use_parallel
from multiprocessing.pool import Pool
import os

logger = logging.getLogger(__name__)


def _concat_outputs(positive_output, negative_output):
    """Concatenate positive and negative outputs, handling tuple (aux) outputs."""
    if isinstance(positive_output, tuple):
        neg_zeros = [negative_output] + [np.zeros_like(negative_output) for _ in range(len(positive_output) - 1)]
        return tuple(np.concatenate([p, n]) for p, n in zip(positive_output, neg_zeros))
    return np.concatenate([positive_output, negative_output])


def literal_eval_array_columns(data_frame: pd.DataFrame, columns: List[Column], pool: Pool = None):
    """Parse list-of-values columns from their CSV string form.

    For each ``Column`` whose ``type`` is one of
    ``FLOAT_ARRAY``, ``INT_ARRAY``, ``INDEXABLE_ARRAY`` or
    ``SEQUENCE``, evaluates the per-row Python literal so the column
    holds actual lists instead of strings. Runs in parallel when a
    multiprocessing ``Pool`` is provided.
    """
    for column in columns:
        if (
            column.type in (IOType.FLOAT_ARRAY, IOType.INT_ARRAY, IOType.INDEXABLE_ARRAY, IOType.SEQUENCE)
            and column.name in data_frame
        ):
            data_frame[column.name] = parallel_literal_eval(data_frame[column.name], pool=pool)


def preprocess_interactions_data_frame(data_frame: pd.DataFrame, project_config: ProjectConfig):
    """Normalise the interactions DataFrame before it enters a Dataset.

    Casts the user / item columns to strings (so they round-trip
    cleanly through the index mapping) and parses any array columns
    listed in the project config from their CSV-literal form. The
    ``available_items_column_name`` column gets the same treatment so
    slate negatives can be sampled directly.

    Mutates and returns ``data_frame``.
    """
    if len(data_frame) == 0:
        return data_frame

    user_col = data_frame[project_config.user_column.name]
    if user_col.dtype != object:
        data_frame[project_config.user_column.name] = user_col.astype(str)

    item_col = data_frame[project_config.item_column.name]
    if item_col.dtype != object:
        data_frame[project_config.item_column.name] = item_col.astype(str)

    all_columns = (
        [
            project_config.user_column,
            project_config.item_column,
            project_config.output_column,
        ]
        + [input_column for input_column in project_config.other_input_columns]
        + [input_column for input_column in project_config.auxiliar_output_columns]
        + [input_column for input_column in project_config.metadata_columns]
    )

    # Check if any array columns actually need parsing
    has_array_columns = any(
        col.type in (IOType.FLOAT_ARRAY, IOType.INT_ARRAY, IOType.INDEXABLE_ARRAY, IOType.SEQUENCE)
        and col.name in data_frame
        for col in all_columns
    )
    needs_available_items = (
        project_config.available_items_column_name
        and project_config.available_items_column_name in data_frame.columns
        and isinstance(data_frame.iloc[0][project_config.available_items_column_name], str)
    )

    if not (has_array_columns or needs_available_items):
        return data_frame

    # One pool shared by every column, and only for large frames: spawning
    # workers costs seconds, more than parsing a few million short lists.
    pool_context = Pool(os.cpu_count()) if use_parallel(len(data_frame)) else contextlib.nullcontext()
    with pool_context as pool:
        literal_eval_array_columns(data_frame, all_columns, pool=pool)

        if needs_available_items:
            data_frame[project_config.available_items_column_name] = parallel_literal_eval(
                data_frame[project_config.available_items_column_name], pool=pool
            )

    return data_frame


def preprocess_metadata_data_frame(
    metadata_data_frame: pd.DataFrame, project_config: ProjectConfig
) -> Dict[str, np.ndarray]:
    """Build the ``{column_name: ndarray}`` lookup for ``metadata_columns``.

    Expects the metadata frame to be already indexed (i.e.
    ``transform_with_indexing`` has rewritten the item column to int
    ids) and the item id column to be contiguous starting at index 3
    (the three reserved slots: unknown, NaN, pad). Stacks each
    metadata column into a numpy array of shape ``(n_items + 3, ...)``
    keyed by item index, with the first 3 rows left as zeros for the
    reserved slots.

    :returns: Mapping from metadata column name to the stacked array,
        looked up at sample time by :class:`InteractionsDataset`.
    """
    NON_UTIL_UID = 3

    metadata_data_frame = metadata_data_frame[
        metadata_data_frame[project_config.item_column.name] != 0
    ]  # Remove unknown

    metadata_data_frame = metadata_data_frame.set_index(project_config.item_column.name, drop=False).sort_index()

    if not (np.arange(NON_UTIL_UID, len(metadata_data_frame.index) + NON_UTIL_UID) == metadata_data_frame.index).all():
        raise ValueError("The item index is not contiguous")

    embeddings_for_metadata: Dict[str, np.ndarray] = {}
    for metadata_column in project_config.metadata_columns:
        emb = metadata_data_frame[metadata_column.name].values.tolist()
        embedding = np.array(emb, dtype=metadata_column.type.dtype)
        # Match the pad dtype to the embedding so np.concatenate doesn't
        # promote a FLOAT_ARRAY (float32) to float64 — which would later
        # crash any nn.Linear that expects its parameter dtype.
        pad = np.zeros((NON_UTIL_UID,) + embedding.shape[1:], dtype=embedding.dtype)
        embedding = np.concatenate((pad, embedding))
        embeddings_for_metadata[metadata_column.name] = embedding

    return embeddings_for_metadata


def _rand_int_except(low: int, high: int, exception: int) -> int:
    if high - low <= 1:
        return low
    while True:
        number = np.random.randint(low, high)
        if number != exception:
            return number


def _choose_except(values: list, exception: Any) -> int:
    filtered = [v for v in values if v != exception]
    if not filtered:
        return values[0] if values else exception
    return random.choice(filtered)


def _choose_from_pool_except(pool: np.ndarray, exception: int) -> int:
    """Uniform draw from ``pool`` that avoids ``exception`` when possible.

    Degenerate pools mirror :func:`_rand_int_except`: an empty pool returns
    ``exception`` and a single-item pool returns that item.
    """
    n = len(pool)
    if n == 0:
        return int(exception)
    if n == 1:
        return int(pool[0])
    for _ in range(32):
        pick = int(pool[np.random.randint(0, n)])
        if pick != exception:
            return pick
    others = pool[pool != exception]
    return int(others[np.random.randint(0, len(others))]) if others.size else int(pool[0])


_MAX_REJECTION_ROUNDS = 32


def _choose_from_pool_except_batch(pool: np.ndarray, exceptions: np.ndarray) -> np.ndarray:
    """Vectorised :func:`_choose_from_pool_except`: one uniform draw per entry of ``exceptions``.

    Draws the whole batch at once and redraws only the entries that hit
    their exception (rejection sampling), so each result is uniform over
    ``pool`` minus that entry's exception, as with the scalar version.
    The rare entries still colliding after a few rounds fall back to it.
    """
    exceptions = np.asarray(exceptions, dtype=np.int64).reshape(-1)
    n = len(pool)
    if n == 0:
        return exceptions.copy()
    if n == 1:
        return np.full(len(exceptions), int(pool[0]), dtype=np.int64)
    picks = np.asarray(pool, dtype=np.int64)[np.random.randint(0, n, size=len(exceptions))]
    for _ in range(_MAX_REJECTION_ROUNDS):
        clash = np.flatnonzero(picks == exceptions)
        if clash.size == 0:
            return picks
        picks[clash] = pool[np.random.randint(0, n, size=clash.size)]
    for i in np.flatnonzero(picks == exceptions):
        picks[i] = _choose_from_pool_except(pool, int(exceptions[i]))
    return picks


def _pack_ragged(rows: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Pack per-row integer lists into ``(padded_matrix, lengths)``, or ``None`` if not integer."""
    lengths = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
    width = int(lengths.max()) if len(lengths) else 0
    matrix = np.zeros((len(rows), max(width, 1)), dtype=np.int64)
    try:
        for i, r in enumerate(rows):
            if len(r):
                matrix[i, : len(r)] = r
    except (TypeError, ValueError):
        return None
    return matrix, lengths


def _choose_row_except_batch(
    matrix: np.ndarray, lengths: np.ndarray, rows: np.ndarray, exceptions: np.ndarray
) -> np.ndarray:
    """Vectorised :func:`_choose_except` over rows of a padded matrix.

    For each ``rows[i]`` draws uniformly among the first ``lengths[rows[i]]``
    entries of that row that differ from ``exceptions[i]`` (duplicates keep
    their weight, as in the scalar version). Rows with no such entry return
    the row's first value, or the exception when the row is empty.
    """
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    exceptions = np.asarray(exceptions, dtype=np.int64).reshape(-1)
    row_len = lengths[rows]
    picks = exceptions.copy()
    todo = np.flatnonzero(row_len > 0)
    for _ in range(_MAX_REJECTION_ROUNDS):
        if todo.size == 0:
            return picks
        cols = (np.random.random_sample(todo.size) * row_len[todo]).astype(np.int64)
        drawn = matrix[rows[todo], cols]
        ok = drawn != exceptions[todo]
        picks[todo[ok]] = drawn[ok]
        todo = todo[~ok]
    for i in todo:
        values = matrix[rows[i], : row_len[i]].tolist()
        picks[i] = _choose_except(values, int(exceptions[i]))
    return picks


def _choice_except(probs: np.ndarray, exception: int) -> Optional[int]:
    """Draw an index from ``probs`` with ``exception``'s mass removed.

    Returns ``None`` when no other index has positive probability, so the
    caller can fall back instead of resampling forever.
    """
    if 0 <= exception < len(probs) and probs[exception] > 0:
        p = probs.copy()
        p[exception] = 0.0
        total = p.sum()
        if total <= 0:
            return None
        p /= total
    else:
        p = probs
    return int(np.random.choice(len(p), p=p))


class InteractionsDataset(Dataset):
    """Vectorised PyTorch ``Dataset`` over the prepared interaction CSV.

    The backbone of every Dataset in this module. On construction it
    pre-packs each declared input column into a contiguous numpy
    array (sized and dtyped per its :class:`IOType`) so that
    ``__getitem__`` becomes O(B) fancy indexing instead of per-row
    pandas access. Returns ``(inputs, output)`` where ``inputs`` is a
    tuple of numpy arrays in :attr:`ProjectConfig.input_columns` order
    plus the per-item ``metadata_columns`` (looked up by item index)
    when an ``embeddings_for_metadata`` mapping is provided.

    Subclasses add negative-sampling logic on top of this; they should
    not need to touch the input packing.

    :param data_frame: Pre-processed interactions DataFrame (returned
        by :func:`preprocess_interactions_data_frame`).
    :param embeddings_for_metadata: ``{column_name: ndarray}`` lookup
        produced by :func:`preprocess_metadata_data_frame`, or ``None``
        when the project does not declare ``metadata_columns``.
    :param project_config: The project's :class:`ProjectConfig`.
    :param index_mapping: ``{column_name: {raw: int_index}}`` used by
        a few negative-sampler variants to draw ids from the catalog.

    The optional ``candidate_feature_lookup`` (a dict, or a 0-/1-arg
    callable returning one) and ``candidate_feature_fn`` keyword arguments
    rewrite item-dependent input columns when a row is reused as a negative
    sample, keeping the features consistent with the swapped item id. See
    :mod:`mars_rec.data.candidate_features`.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[Any, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        *args,
        **kwargs,
    ) -> None:
        # Pop the per-candidate feature params *before* they can leak into a
        # subclass kwargs forward (unrecognised kwargs raise). ``None`` (the
        # default) makes the override a no-op.
        self._candidate_feature_lookup_raw = kwargs.pop(LOOKUP_PARAM, None)
        self._candidate_feature_lookup: Optional[Dict[str, Any]] = None
        self._candidate_feature_fn = validate_candidate_feature_fn(kwargs.pop(FN_PARAM, None))

        self._project_config = project_config
        self._index_mapping = index_mapping
        self._input_columns: List[Column] = project_config.input_columns
        input_column_names = [input_column.name for input_column in self._input_columns]

        # Pre-resolve column→tuple-position so the override is O(1) per call.
        self._input_col_index_by_name: Dict[str, int] = {col.name: i for i, col in enumerate(self._input_columns)}

        if project_config.item_is_input:
            self._item_input_index = self._input_columns.index(project_config.item_column)

        auxiliar_output_column_names = [
            auxiliar_output_column.name for auxiliar_output_column in project_config.auxiliar_output_columns
        ]
        self._data_frame = data_frame[
            list(
                set(
                    input_column_names + [project_config.output_column.name] + auxiliar_output_column_names
                ).intersection(data_frame.columns)
            )
        ]
        self._embeddings_for_metadata = embeddings_for_metadata

        # Pre-pack each column to its target dtype/shape ONCE up front so
        # __getitem__ becomes O(B) numpy fancy indexing instead of pandas
        # iloc + per-element ``np.array(...)`` per batch. Saved O(N) times
        # the per-batch loop overhead — measured 2-3× per-batch speedup on
        # a 37-column schema with ~83k train rows.
        self._packed_inputs: Dict[str, np.ndarray] = {}
        for col in self._input_columns:
            if col.name in self._data_frame.columns:
                self._packed_inputs[col.name] = self._convert_dtype(
                    self._data_frame[col.name].values,
                    col.type,
                )
        out_col = project_config.output_column
        self._packed_output: np.ndarray = self._convert_dtype(
            self._data_frame[out_col.name].values,
            out_col.type,
        )
        self._packed_aux_outputs: List[Tuple[Column, np.ndarray]] = [
            (
                col,
                self._convert_dtype(
                    self._data_frame[col.name].values,
                    col.type,
                ),
            )
            for col in project_config.auxiliar_output_columns
            if col.name in self._data_frame.columns
        ]

    # ---------------------------------------------------------------------
    # Per-candidate feature override (used by negative samplers)
    # ---------------------------------------------------------------------
    def _resolve_candidate_lookup(self) -> Optional[Dict[str, Any]]:
        """Materialise ``candidate_feature_lookup`` on first use; cache it.

        See :func:`mars_rec.data.candidate_features.resolve_candidate_lookup`.
        """
        if self._candidate_feature_lookup is None:
            self._candidate_feature_lookup = resolve_candidate_lookup(
                self._candidate_feature_lookup_raw, self._index_mapping
            )
        return self._candidate_feature_lookup

    def _apply_candidate_features(
        self,
        input_tuple: Tuple[np.ndarray, ...],
        neg_item_indices: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """Rewrite item-dependent columns of ``input_tuple`` for the negative items.

        Applies the project's ``candidate_feature_lookup`` (direct lookups) and
        ``candidate_feature_fn`` (computed features) through
        :func:`mars_rec.data.candidate_features.apply_candidate_features`, the
        same helper the evaluation and inference paths use. A no-op when
        neither is configured.
        """
        lookup = self._resolve_candidate_lookup()
        if not lookup and self._candidate_feature_fn is None:
            return input_tuple
        col_idx = self._input_col_index_by_name
        input_list = list(input_tuple)
        updates = apply_candidate_features(
            {name: input_list[i] for name, i in col_idx.items() if i < len(input_list)},
            neg_item_indices,
            self._input_columns,
            lookup,
            self._candidate_feature_fn,
            item_column_name=self._project_config.item_column.name,
        )
        for name, values in updates.items():
            input_list[col_idx[name]] = values
        return tuple(input_list)

    def __len__(self) -> int:
        return self._data_frame.shape[0]

    def _convert_dtype(self, value: np.ndarray, type: IOType) -> np.ndarray:
        return convert_dtype(value, type)

    def __getitem__(
        self, indices: Union[int, List[int], slice]
    ) -> Tuple[Tuple[np.ndarray, ...], Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            # Numpy fancy index doesn't accept slice → materialise a range.
            indices = list(range(len(self))[indices])
        idx = np.asarray(indices, dtype=np.int64)

        # Vectorised fetch from the pre-packed column arrays. Order follows
        # ``_input_columns`` so the returned tuple positions match the
        # model's ``forward`` signature.
        inputs = tuple(
            self._packed_inputs[col.name][idx] for col in self._input_columns if col.name in self._packed_inputs
        )

        if self._project_config.item_is_input and self._embeddings_for_metadata is not None:
            item_indices = inputs[self._item_input_index]
            inputs += tuple(
                self._embeddings_for_metadata[column.name][item_indices]
                for column in self._project_config.metadata_columns
                if column.name not in self._data_frame.columns
            )

        output: Any = self._packed_output[idx]
        if self._packed_aux_outputs:
            output = (output,) + tuple(arr[idx] for _, arr in self._packed_aux_outputs)

        return inputs, output

    def _relookup_metadata(self, input_list: list, new_item_indices: np.ndarray) -> None:
        """Re-look-up metadata columns for swapped item indices (in-place).

        After negative sampling replaces the item index, the metadata
        slots (appended after packed-input columns) still hold the
        original positive item's values.  This method overwrites them
        with the correct values for ``new_item_indices``.
        """
        if self._embeddings_for_metadata is None:
            return
        n_packed = sum(1 for c in self._input_columns if c.name in self._packed_inputs)
        offset = 0
        for col in self._project_config.metadata_columns:
            if col.name not in self._data_frame.columns:
                input_list[n_packed + offset] = self._embeddings_for_metadata[col.name][new_item_indices]
                offset += 1

    # ------------------------------------------------------------------
    # Negative-sampling helpers shared by the subclasses
    # ------------------------------------------------------------------

    @property
    def _negative_item_pool(self) -> np.ndarray:
        """Real item indices (``<= _max_item_idx``) that may be drawn as negatives.

        Excludes the reserved index-mapping slots (unknown, NaN, pad; see
        :func:`~mars_rec.utils.index_mapping.real_indices`), so a negative is
        always a real catalog item. Falls back to the legacy
        ``[0, _max_item_idx]`` range when the item mapping has no real
        entries (e.g. a hand-built mapping). Requires ``_max_item_idx``.
        """
        pool = self.__dict__.get("_negative_item_pool_cache")
        if pool is None:
            max_idx = int(self._max_item_idx)
            item_map = self._index_mapping.get(self._project_config.item_column.name) or {}
            pool = real_indices(item_map)
            pool = pool[pool <= max_idx]
            if pool.size == 0:
                pool = np.arange(0, max_idx + 1, dtype=np.int64)
            self._negative_item_pool_cache = pool
        return pool

    def _random_negative(self, exclude: int) -> int:
        """Uniform negative from :attr:`_negative_item_pool`, avoiding ``exclude``."""
        return _choose_from_pool_except(self._negative_item_pool, int(exclude))

    def _random_negatives(self, excludes: np.ndarray) -> np.ndarray:
        """Batch :meth:`_random_negative`: one negative per entry of ``excludes``."""
        return _choose_from_pool_except_batch(self._negative_item_pool, excludes)

    def _resample_except(self, probs: np.ndarray, exclude: int) -> int:
        """Redraw from ``probs`` avoiding ``exclude``; uniform pool fallback when it holds all the mass."""
        pick = _choice_except(probs, exclude)
        return self._random_negative(exclude) if pick is None else pick

    def _restrict_to_pool(self, weights: np.ndarray) -> np.ndarray:
        """Normalise ``weights`` (indexed by item idx) into a distribution over the pool.

        Mass outside :attr:`_negative_item_pool` (e.g. reserved slots) is
        dropped; when nothing in the pool has positive weight the result is
        uniform over the pool.
        """
        in_pool = np.zeros(len(weights), dtype=bool)
        pool = self._negative_item_pool
        in_pool[pool[pool < len(weights)]] = True
        w = np.where(in_pool, np.asarray(weights, dtype=np.float64), 0.0)
        total = w.sum()
        if total <= 0:
            w = in_pool.astype(np.float64)
            total = w.sum()
        return w / total


class InteractionsWithNegativeItemGenerationDataset(InteractionsDataset):
    """Uniform-random negative sampling over the full catalog.

    Extends :class:`InteractionsDataset` by stretching ``__len__`` to
    include negative slots and producing them in ``__getitem__`` by
    copying a positive row and replacing the item id with a uniformly
    random real item (see :attr:`InteractionsDataset._negative_item_pool`),
    excluding the original positive. The ``negative_proportion`` controls the share of
    negatives in the materialised epoch.

    This is the default choice for "BCE + implicit feedback" projects
    — see ``examples/movielens`` and ``examples/lastfm``.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping, *args, **kwargs)
        self._negative_proportion = negative_proportion
        self._max_item_idx = data_frame[project_config.item_column.name].max()

    def __len__(self) -> int:
        return super().__len__() + int((1 / (1 - self._negative_proportion) - 1) * super().__len__())

    def __getitem__(self, indices: Union[int, List[int], slice]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()

        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        positive_input, positive_output = super().__getitem__(positive_indices)

        if num_of_negatives > 0:
            sample_positive_indices = list(np.random.randint(0, n, size=num_of_negatives))

            negative_input, _ = super().__getitem__(sample_positive_indices)
            negative_output = self._convert_dtype(np.zeros(num_of_negatives), self._project_config.output_column.type)

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = self._random_negatives(negative_input[self._item_input_index])
            self._relookup_metadata(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(
                    np.concatenate([positive_array, negative_array])
                    for positive_array, negative_array in zip(positive_input, negative_input)
                )
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output


class InteractionsWithNegativeItemGenerationByAvailableItemsDataset(InteractionsDataset):
    """Slate-restricted negative sampling.

    Like :class:`InteractionsWithNegativeItemGenerationDataset`, but
    negatives are drawn from the per-row candidate slate stored in the
    ``available_items_column_name`` column instead of the full
    catalog. Matches the "leave-one-out + 99 random negatives" eval
    convention used by the NCF / MIND papers — see
    ``examples/movielens100k_slate`` for a worked example.

    The constructor asserts the slate column is present and re-indexes
    its values once via the item index_mapping so per-call sampling is
    a cheap ``random.choice`` against pre-indexed ids.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        *args,
        **kwargs,
    ) -> None:
        assert project_config.available_items_column_name in data_frame
        self._available_items = (
            data_frame[project_config.available_items_column_name]
            .map(functools.partial(map_array, mapping=index_mapping[project_config.item_column.name]))
            .values
        )

        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping, *args, **kwargs)
        self._negative_proportion = negative_proportion

    def __len__(self) -> int:
        return super().__len__() + int((1 / (1 - self._negative_proportion) - 1) * super().__len__())

    def _choose_available_negatives(self, rows: np.ndarray, positives: np.ndarray) -> np.ndarray:
        """One negative per row, drawn from that row's available items minus its positive.

        Packs the per-row lists into a padded matrix on first use so a batch
        is sampled with array operations instead of a Python loop per row.
        """
        packed = self.__dict__.get("_available_items_packed")
        if packed is None:
            packed = _pack_ragged(self._available_items) or False
            self._available_items_packed = packed
        if packed is False:  # non-integer items: keep the per-row path
            return np.array([_choose_except(self._available_items[r], p) for r, p in zip(rows, positives)])
        matrix, lengths = packed
        return _choose_row_except_batch(matrix, lengths, rows, positives)

    def __getitem__(self, indices: Union[int, List[int], slice]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()

        positive_indices = [index for index in indices if index < n]
        negative_indices = [index - (index // n) * n for index in indices if index >= n]

        positive_input, positive_output = super().__getitem__(positive_indices)

        if negative_indices:
            negative_input, _ = super().__getitem__(negative_indices)

            negative_output = self._convert_dtype(
                np.zeros(len(negative_indices)), self._project_config.output_column.type
            )

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = self._choose_available_negatives(
                np.asarray(negative_indices, dtype=np.int64), negative_input[self._item_input_index]
            )
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(
                    np.concatenate([positive_array, negative_array])
                    for positive_array, negative_array in zip(positive_input, negative_input)
                )
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output


# =============================================================================
# Strategy 1: Popularity-Based Negative Sampling
# =============================================================================


class InteractionsWithPopularityBasedNegativeSamplingDataset(InteractionsDataset):
    """Negative sampling where items are sampled proportionally to their popularity.

    Popular items that the user did NOT interact with are harder negatives
    than obscure items, forcing the model to learn finer distinctions.

    Args:
        smoothing_factor: Exponent applied to item frequencies.
            1.0 = pure popularity, 0.0 = uniform, 0.5 = smoothed (default).
        popularity_column: Optional column name in the data frame containing
            external popularity scores per item. When provided, these scores
            are used instead of interaction frequency counts.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        smoothing_factor: float = 0.5,
        popularity_column: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping, *args, **kwargs)
        self._negative_proportion = negative_proportion
        self._max_item_idx = data_frame[project_config.item_column.name].max()

        freq = np.zeros(self._max_item_idx + 1, dtype=np.float64)

        if popularity_column and popularity_column in data_frame.columns:
            # Use external popularity scores aggregated per item
            item_col = project_config.item_column.name
            item_pop = data_frame.groupby(item_col)[popularity_column].first()
            for item_idx, pop_score in item_pop.items():
                if 0 <= item_idx <= self._max_item_idx:
                    freq[item_idx] = max(0.0, float(pop_score))
        else:
            # Fallback: interaction frequency counts
            item_counts = data_frame[project_config.item_column.name].value_counts()
            for item_idx, count in item_counts.items():
                if 0 <= item_idx <= self._max_item_idx:
                    freq[item_idx] = count

        self._item_probabilities = self._restrict_to_pool(np.power(freq, smoothing_factor))

    def __len__(self) -> int:
        return super().__len__() + int((1 / (1 - self._negative_proportion) - 1) * super().__len__())

    def __getitem__(self, indices: Union[int, List[int], slice]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()
        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        positive_input, positive_output = super().__getitem__(positive_indices)

        if num_of_negatives > 0:
            sample_positive_indices = list(np.random.randint(0, n, size=num_of_negatives))
            negative_input, _ = super().__getitem__(sample_positive_indices)
            negative_output = self._convert_dtype(np.zeros(num_of_negatives), self._project_config.output_column.type)

            # Sample replacement items using popularity distribution
            original_items = negative_input[self._item_input_index]
            new_items = np.random.choice(
                len(self._item_probabilities),
                size=num_of_negatives,
                p=self._item_probabilities,
            )
            # Ensure sampled item differs from the original positive item
            for i in range(num_of_negatives):
                if new_items[i] == original_items[i]:
                    new_items[i] = self._resample_except(self._item_probabilities, int(original_items[i]))

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = new_items.astype(np.int64)
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(np.concatenate([p, n_]) for p, n_ in zip(positive_input, negative_input))
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output


# =============================================================================
# Strategy 2: Hard Negative Mining
# =============================================================================


class InteractionsWithHardNegativeMiningDataset(InteractionsDataset):
    """Negative sampling using pre-computed hard negatives.

    Hard negatives are items that a scoring model ranks highly but the user
    did NOT interact with. They must be pre-computed externally (e.g. by
    running inference with a previous model checkpoint) and stored in a column
    of the data frame.

    Falls back to random negatives when hard negatives are unavailable.

    Args:
        hard_negative_column: Column name containing lists of hard negative item indices.
        hard_negative_ratio: Fraction of negatives that should be hard (rest are random).
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        hard_negative_column: str = "hard_negatives",
        hard_negative_ratio: float = 0.5,
        *args,
        **kwargs,
    ) -> None:
        self._hard_negative_ratio = hard_negative_ratio
        self._has_hard_negatives = hard_negative_column in data_frame.columns

        if self._has_hard_negatives:
            self._hard_negatives = data_frame[hard_negative_column].values
            if len(self._hard_negatives) > 0 and isinstance(self._hard_negatives[0], str):
                import ast

                self._hard_negatives = np.array([ast.literal_eval(v) for v in self._hard_negatives], dtype=object)

        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping, *args, **kwargs)
        self._negative_proportion = negative_proportion
        self._max_item_idx = data_frame[project_config.item_column.name].max()

    def __len__(self) -> int:
        return super().__len__() + int((1 / (1 - self._negative_proportion) - 1) * super().__len__())

    def __getitem__(self, indices: Union[int, List[int], slice]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()
        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        positive_input, positive_output = super().__getitem__(positive_indices)

        if num_of_negatives > 0:
            sample_positive_indices = list(np.random.randint(0, n, size=num_of_negatives))
            negative_input, _ = super().__getitem__(sample_positive_indices)
            negative_output = self._convert_dtype(np.zeros(num_of_negatives), self._project_config.output_column.type)

            original_items = negative_input[self._item_input_index]
            new_items = np.empty(num_of_negatives, dtype=np.int64)

            num_hard = int(num_of_negatives * self._hard_negative_ratio) if self._has_hard_negatives else 0

            # Hard negatives
            for i in range(num_hard):
                src_idx = sample_positive_indices[i]
                candidates = self._hard_negatives[src_idx]
                if len(candidates) > 0:
                    new_items[i] = _choose_except(list(candidates), original_items[i])
                else:
                    new_items[i] = self._random_negative(original_items[i])

            # Random negatives for the rest
            new_items[num_hard:] = self._random_negatives(original_items[num_hard:])

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = new_items
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(np.concatenate([p, n_]) for p, n_ in zip(positive_input, negative_input))
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output


# =============================================================================
# Strategy 3: In-Batch Negatives
# =============================================================================


class InteractionsWithInBatchNegativesDataset(InteractionsDataset):
    """Uses positive items from other samples in the same batch as negatives.

    Extremely efficient: zero extra memory or computation for generating negatives.
    Works best with two-tower / embedding models.

    The __getitem__ returns an extra output: the full batch of item indices,
    so the training loop / loss function can build the negative pairs.

    With `include_random_negatives=True`, also adds traditional random negatives
    alongside the in-batch signal.

    Args:
        include_random_negatives: Whether to also add random negatives.
        random_negative_proportion: Proportion of random negatives (only if above is True).
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        include_random_negatives: bool = False,
        random_negative_proportion: float = 0.2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping, *args, **kwargs)
        self._include_random_negatives = include_random_negatives
        self._random_negative_proportion = random_negative_proportion
        self._max_item_idx = data_frame[project_config.item_column.name].max()

    def __len__(self) -> int:
        base_len = super().__len__()
        if self._include_random_negatives:
            return base_len + int((1 / (1 - self._random_negative_proportion) - 1) * base_len)
        return base_len

    def __getitem__(
        self, indices: Union[int, List[int], slice]
    ) -> Tuple[Tuple[np.ndarray, ...], Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()
        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)

        positive_input, positive_output = super().__getitem__(positive_indices)

        # Expose in-batch item indices as an auxiliary output for the loss function
        if self._project_config.item_is_input:
            in_batch_items = positive_input[self._item_input_index].copy()
        else:
            in_batch_items = np.array([], dtype=np.int64)

        if num_of_negatives > 0 and self._include_random_negatives:
            sample_positive_indices = list(np.random.randint(0, n, size=num_of_negatives))
            negative_input, _ = super().__getitem__(sample_positive_indices)
            negative_output = self._convert_dtype(np.zeros(num_of_negatives), self._project_config.output_column.type)

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = self._random_negatives(negative_input[self._item_input_index])
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(np.concatenate([p, n_]) for p, n_ in zip(positive_input, negative_input))
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        # Return in_batch_items as a tuple with the main output
        return input_, (output, in_batch_items)


# =============================================================================
# Strategy 4: Time-Aware Negative Sampling
# =============================================================================


class InteractionsWithTimeAwareNegativeSamplingDataset(InteractionsDataset):
    """Negative sampling restricted to items that existed at interaction time.

    Prevents temporal leakage: an item launched in 2025 cannot be a negative
    for a 2023 interaction. Requires a timestamp column in the interactions
    data frame AND an item-to-first-seen mapping.

    Args:
        item_first_seen: Dict mapping item index -> first-seen timestamp (numeric).
            If None, it is inferred from the data frame as the minimum timestamp per item.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        item_first_seen: Optional[
            Union[Dict[int, float], Callable[[Dict[str, Dict[Any, int]]], Dict[int, float]]]
        ] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping, *args, **kwargs)
        self._negative_proportion = negative_proportion
        self._max_item_idx = data_frame[project_config.item_column.name].max()

        # Store timestamps for each interaction
        ts_col = project_config.timestamp_column_name
        if ts_col in data_frame.columns:
            self._timestamps = data_frame[ts_col].values
        else:
            self._timestamps = None

        # Resolve a callable factory the same way ``candidate_feature_lookup``
        # is resolved upstream: 0-arg (() -> dict) or 1-arg
        # ((index_mapping) -> dict). Lets configs defer the metadata fetch
        # to dataset-construction time instead of import time.
        if callable(item_first_seen):
            try:
                sig = inspect.signature(item_first_seen)
                arity = len(
                    [
                        p
                        for p in sig.parameters.values()
                        if p.kind
                        in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                        and p.default is inspect.Parameter.empty
                    ]
                )
            except (TypeError, ValueError):
                arity = 0
            item_first_seen = item_first_seen(index_mapping) if arity >= 1 else item_first_seen()

        # Build item_first_seen mapping
        if item_first_seen is not None:
            self._item_first_seen = item_first_seen
        elif self._timestamps is not None:
            item_col = project_config.item_column.name
            self._item_first_seen = data_frame.groupby(item_col)[ts_col].min().to_dict()
        else:
            self._item_first_seen = None

        # Pre-compute sorted arrays for fast filtering
        if self._item_first_seen is not None:
            max_idx = self._max_item_idx + 1
            self._first_seen_array = np.full(max_idx, np.inf)
            for item_idx, ts in self._item_first_seen.items():
                if 0 <= item_idx < max_idx:
                    self._first_seen_array[item_idx] = ts

            # Sort items by first_seen for efficient binary-search filtering
            self._items_sorted_by_time = np.argsort(self._first_seen_array)
            self._first_seen_sorted = self._first_seen_array[self._items_sorted_by_time]

    def __len__(self) -> int:
        return super().__len__() + int((1 / (1 - self._negative_proportion) - 1) * super().__len__())

    def _sample_item_before(self, timestamp: float, exclude: int) -> int:
        """Sample a random item whose first_seen <= timestamp, excluding `exclude`."""
        if self._item_first_seen is None:
            return self._random_negative(exclude)

        # Binary search for the rightmost item with first_seen <= timestamp
        pos = np.searchsorted(self._first_seen_sorted, timestamp, side="right")
        if pos == 0:
            return self._random_negative(exclude)

        eligible = self._items_sorted_by_time[:pos]
        if len(eligible) <= 1:
            return eligible[0] if len(eligible) == 1 else exclude

        # Sample with rejection for exclude
        for _ in range(10):
            chosen = eligible[np.random.randint(0, len(eligible))]
            if chosen != exclude:
                return int(chosen)
        return int(eligible[0]) if eligible[0] != exclude else int(eligible[-1])

    def __getitem__(self, indices: Union[int, List[int], slice]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()
        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        positive_input, positive_output = super().__getitem__(positive_indices)

        if num_of_negatives > 0:
            sample_positive_indices = list(np.random.randint(0, n, size=num_of_negatives))
            negative_input, _ = super().__getitem__(sample_positive_indices)
            negative_output = self._convert_dtype(np.zeros(num_of_negatives), self._project_config.output_column.type)

            original_items = negative_input[self._item_input_index]

            if self._timestamps is not None:
                timestamps = self._timestamps[sample_positive_indices]
                new_items = np.array(
                    [self._sample_item_before(ts, item_idx) for ts, item_idx in zip(timestamps, original_items)],
                    dtype=np.int64,
                )
            else:
                new_items = self._random_negatives(original_items)

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = new_items
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(np.concatenate([p, n_]) for p, n_ in zip(positive_input, negative_input))
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output


# =============================================================================
# Strategy 5: Curriculum Negative Sampling
# =============================================================================


class InteractionsWithCurriculumNegativeSamplingDataset(InteractionsDataset):
    """Progressively harder negative sampling during training (Curriculum Learning).

    Starts with easy (uniform random) negatives and gradually shifts toward
    popularity-based (harder) negatives as training progresses. The transition
    is controlled by calling `set_epoch()`.

    Args:
        total_epochs: Total number of training epochs (used to compute schedule).
        warmup_epochs: Number of initial epochs using purely random negatives.
        smoothing_factor: Popularity exponent at full difficulty (see PopularityBased).
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        total_epochs: int = 20,
        warmup_epochs: int = 3,
        smoothing_factor: float = 0.75,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping, *args, **kwargs)
        self._negative_proportion = negative_proportion
        self._max_item_idx = data_frame[project_config.item_column.name].max()
        self._total_epochs = total_epochs
        self._warmup_epochs = warmup_epochs
        self._current_epoch = 0
        self._difficulty = 0.0  # 0.0 = all random, 1.0 = all popularity-based

        # Build popularity distribution
        item_counts = data_frame[project_config.item_column.name].value_counts()
        freq = np.zeros(self._max_item_idx + 1, dtype=np.float64)
        for item_idx, count in item_counts.items():
            if 0 <= item_idx <= self._max_item_idx:
                freq[item_idx] = count

        self._popularity_probs = self._restrict_to_pool(np.power(freq, smoothing_factor))

        # Uniform distribution over the real items
        self._uniform_probs = self._restrict_to_pool(np.ones(self._max_item_idx + 1, dtype=np.float64))

        self._update_sampling_distribution()

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch to adjust sampling difficulty."""
        self._current_epoch = epoch
        if epoch < self._warmup_epochs:
            self._difficulty = 0.0
        else:
            progress = (epoch - self._warmup_epochs) / max(1, self._total_epochs - self._warmup_epochs)
            self._difficulty = min(1.0, progress)
        self._update_sampling_distribution()

    def _update_sampling_distribution(self) -> None:
        """Blend uniform and popularity distributions based on current difficulty."""
        d = self._difficulty
        self._current_probs = (1 - d) * self._uniform_probs + d * self._popularity_probs
        # Re-normalize to account for floating point
        self._current_probs /= self._current_probs.sum()

    def __len__(self) -> int:
        return super().__len__() + int((1 / (1 - self._negative_proportion) - 1) * super().__len__())

    def __getitem__(self, indices: Union[int, List[int], slice]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()
        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        positive_input, positive_output = super().__getitem__(positive_indices)

        if num_of_negatives > 0:
            sample_positive_indices = list(np.random.randint(0, n, size=num_of_negatives))
            negative_input, _ = super().__getitem__(sample_positive_indices)
            negative_output = self._convert_dtype(np.zeros(num_of_negatives), self._project_config.output_column.type)

            original_items = negative_input[self._item_input_index]
            new_items = np.random.choice(
                len(self._current_probs),
                size=num_of_negatives,
                p=self._current_probs,
            )
            for i in range(num_of_negatives):
                if new_items[i] == original_items[i]:
                    new_items[i] = self._resample_except(self._current_probs, int(original_items[i]))

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = new_items.astype(np.int64)
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(np.concatenate([p, n_]) for p, n_ in zip(positive_input, negative_input))
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output


# =============================================================================
# Strategy 6: Embedding-Based ANN Hard Negative Sampling
# =============================================================================


class InteractionsWithEmbeddingHardNegativeDataset(InteractionsDataset):
    """Hard negatives drawn from a pre-computed top-K nearest-neighbour table.

    Inputs:
        ``neighbors_path`` (str): pickle file containing
        ``{item_uuid: [neighbor_uuid_1, ...]}`` produced offline (e.g. by
        a CF-BPR / embedding kNN job).

    At construction time the uuid lists are translated to framework idx via
    ``index_mapping[item_column]`` and stored as a list-of-numpy arrays
    indexed by item idx.

    Sampling: for each negative slot, look up the *positive item idx*'s
    neighbour list and sample one uniformly. With probability
    ``uniform_fallback_ratio`` the sample is a uniform random idx instead —
    keeps a regularising stream of easy negatives so the loss does not
    collapse to "tell ANN-near apart".

    Args:
        neighbors_path: path to the pre-computed neighbour pickle.
        uniform_fallback_ratio: fraction of negatives drawn uniformly.
        top_k_use: maximum neighbours per item to consider (slice).
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        neighbors_path: Optional[str] = None,
        uniform_fallback_ratio: float = 0.2,
        top_k_use: int = 50,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping, *args, **kwargs)
        self._negative_proportion = negative_proportion
        self._uniform_fallback_ratio = float(uniform_fallback_ratio)
        self._top_k_use = int(top_k_use)
        self._max_item_idx = int(data_frame[project_config.item_column.name].max())

        if not neighbors_path:
            raise ValueError(
                "InteractionsWithEmbeddingHardNegativeDataset requires `neighbors_path` (path to a top-K NN pickle)."
            )
        if not os.path.isfile(neighbors_path):
            raise FileNotFoundError(f"Neighbour matrix not found at {neighbors_path}. Run the pre-compute step first.")

        import pickle as _pickle

        with open(neighbors_path, "rb") as f:
            uuid_to_neighbors: Dict[Any, List[Any]] = _pickle.load(f)

        item_idx_map = index_mapping.get(project_config.item_column.name, {}) or {}
        self._neighbors_by_idx: List[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(self._max_item_idx + 1)]
        matched = 0
        for uuid, neigh_uuids in uuid_to_neighbors.items():
            if uuid not in item_idx_map:
                continue
            src_idx = int(item_idx_map[uuid])
            if not (0 <= src_idx <= self._max_item_idx):
                continue
            neigh_idx = [
                int(item_idx_map[u])
                for u in neigh_uuids[: self._top_k_use]
                if u in item_idx_map and int(item_idx_map[u]) > 1
            ]
            if neigh_idx:
                self._neighbors_by_idx[src_idx] = np.asarray(neigh_idx, dtype=np.int64)
                matched += 1
        logger.info(
            "EmbeddingHardNegative: matched %d/%d items in index_mapping (top_k_use=%d)",
            matched,
            len(uuid_to_neighbors),
            self._top_k_use,
        )

    def __len__(self) -> int:
        return super().__len__() + int((1 / (1 - self._negative_proportion) - 1) * super().__len__())

    def __getitem__(self, indices: Union[int, List[int], slice]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()
        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        positive_input, positive_output = super().__getitem__(positive_indices)

        if num_of_negatives > 0:
            sample_positive_indices = list(np.random.randint(0, n, size=num_of_negatives))
            negative_input, _ = super().__getitem__(sample_positive_indices)
            negative_output = self._convert_dtype(np.zeros(num_of_negatives), self._project_config.output_column.type)

            original_items = negative_input[self._item_input_index]
            new_items = np.empty(num_of_negatives, dtype=np.int64)
            n_uniform = int(num_of_negatives * self._uniform_fallback_ratio)

            for i in range(num_of_negatives):
                src = int(original_items[i])
                neighbours = (
                    self._neighbors_by_idx[src] if 0 <= src <= self._max_item_idx else np.empty(0, dtype=np.int64)
                )
                want_uniform = (i < n_uniform) or (neighbours.size == 0)
                if want_uniform:
                    new_items[i] = self._random_negative(src)
                else:
                    pick = int(np.random.choice(neighbours))
                    if pick == src and neighbours.size > 1:
                        pick = int(np.random.choice(neighbours))
                    new_items[i] = pick

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = new_items
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(np.concatenate([p, n_]) for p, n_ in zip(positive_input, negative_input))
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output


# =============================================================================
# Strategy 7: Goal-Aware (CandidateProvider) Negative Sampling
# =============================================================================


class InteractionsWithCandidateProviderNegativeDataset(InteractionsDataset):
    """Negatives sampled from the project's CandidateProvider eligible pool.

    Consults ``project_config.candidate_provider_factory`` once at __init__
    to build a provider, and per row at sample time to obtain a per-row
    eligible item-idx vector from which the negative is drawn uniformly.

    Falls back to uniform-from-vocab when the eligible pool is empty.
    Designed to align the train-time negative distribution with the
    submission-time eligibility filter.

    Args:
        provider_extras: extra dict forwarded to the factory (e.g. lookups).
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        provider_extras: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping, *args, **kwargs)
        self._negative_proportion = negative_proportion
        self._max_item_idx = int(data_frame[project_config.item_column.name].max())
        self._df_for_rows = data_frame.reset_index(drop=True)

        from mars_rec.filtering import resolve_candidate_provider

        self._provider = resolve_candidate_provider(
            project_config,
            index_mapping,
            provider_extras or {},
        )
        if self._provider is None:
            raise ValueError(
                "InteractionsWithCandidateProviderNegativeDataset requires the "
                "project_config to declare a candidate_provider_factory (or a "
                "legacy available_items_column_name)."
            )

        item_map = index_mapping.get(project_config.item_column.name, {}) or {}
        self._full_catalog = real_indices(item_map)

    def __len__(self) -> int:
        return super().__len__() + int((1 / (1 - self._negative_proportion) - 1) * super().__len__())

    def __getitem__(self, indices: Union[int, List[int], slice]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()
        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        positive_input, positive_output = super().__getitem__(positive_indices)

        if num_of_negatives > 0:
            sample_positive_indices = list(np.random.randint(0, n, size=num_of_negatives))
            negative_input, _ = super().__getitem__(sample_positive_indices)
            negative_output = self._convert_dtype(np.zeros(num_of_negatives), self._project_config.output_column.type)

            original_items = negative_input[self._item_input_index]
            new_items = np.empty(num_of_negatives, dtype=np.int64)

            for i, src_pos_idx in enumerate(sample_positive_indices):
                row = self._df_for_rows.iloc[int(src_pos_idx)].to_dict()
                eligible = self._provider.get_candidates(row, self._full_catalog)
                if eligible is None or len(eligible) == 0:
                    new_items[i] = self._random_negative(int(original_items[i]))
                    continue
                pick = int(np.random.choice(eligible))
                if pick == int(original_items[i]) and len(eligible) > 1:
                    pick = int(np.random.choice(eligible))
                new_items[i] = pick

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = new_items
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(np.concatenate([p, n_]) for p, n_ in zip(positive_input, negative_input))
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output


# =============================================================================
# Strategy 8: Dynamic Negative Mining (DNS — score-based hard negatives)
# =============================================================================


class InteractionsWithDynamicNegativeMiningDataset(InteractionsDataset):
    """Score-driven hard negatives: for each negative slot, draw K uniform
    candidates and keep the one with the highest *current model* score.

    The model is provided lazily via ``set_model(model, device)`` (called
    by the training loop once the LightningModule is built). Until then,
    sampling falls back to uniform — equivalent to a warmup phase.

    Args:
        candidates_per_slot: number of uniform candidates evaluated per
            negative slot. Larger K → harder negatives, more compute.
        warmup_steps: steps to keep using uniform fallback before scoring.

    Note: this sampler increases iteration time linearly with
    ``candidates_per_slot`` — start with K=8 and tune from there.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        candidates_per_slot: int = 8,
        warmup_steps: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping, *args, **kwargs)
        self._negative_proportion = negative_proportion
        self._max_item_idx = int(data_frame[project_config.item_column.name].max())
        self._candidates_per_slot = max(1, int(candidates_per_slot))
        self._warmup_steps = int(warmup_steps)
        self._step_counter = 0
        self._model = None
        self._device = "cpu"

    def set_model(self, model, device: str = "cpu") -> None:
        """Inject the live model so __getitem__ can compute candidate scores."""
        self._model = model
        self._device = device

    def __len__(self) -> int:
        return super().__len__() + int((1 / (1 - self._negative_proportion) - 1) * super().__len__())

    def _draw_candidates(
        self,
        sample_positive_indices: List[int],
        original_items: np.ndarray,
        K: int,
    ) -> np.ndarray:
        """Draw ``K`` candidate item indices per negative slot.

        Default implementation: uniform-random over the catalog (excluding
        the row's positive item). Subclasses override to draw from a
        CandidateProvider, ANN neighbours, etc.

        Args:
            sample_positive_indices: indices of the source positive rows
                (one per negative slot). Used by overrides that need the
                row context.
            original_items: ``[num_negatives]`` int64 — the positive item
                idx per slot, excluded from the candidate pool.
            K: number of candidates to draw per slot.

        Returns:
            ``[num_negatives, K]`` int64 array of candidate item indices.
        """
        num_negatives = len(original_items)
        cand_items = np.empty((num_negatives, K), dtype=np.int64)
        for i, it in enumerate(original_items):
            src = int(it)
            for k in range(K):
                cand_items[i, k] = self._random_negative(src)
        return cand_items

    def __getitem__(self, indices: Union[int, List[int], slice]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()
        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        positive_input, positive_output = super().__getitem__(positive_indices)

        if num_of_negatives > 0:
            sample_positive_indices = list(np.random.randint(0, n, size=num_of_negatives))
            negative_input, _ = super().__getitem__(sample_positive_indices)
            negative_output = self._convert_dtype(np.zeros(num_of_negatives), self._project_config.output_column.type)

            original_items = negative_input[self._item_input_index]
            self._step_counter += 1
            use_dns = self._model is not None and self._step_counter > self._warmup_steps

            if not use_dns:
                # Warmup: 1 candidate per slot, no scoring.
                draws = self._draw_candidates(
                    sample_positive_indices,
                    original_items,
                    K=1,
                )
                new_items = draws[:, 0]
            else:
                K = self._candidates_per_slot
                cand_items = self._draw_candidates(
                    sample_positive_indices,
                    original_items,
                    K=K,
                )

                tiled_input = []
                for col_arr in negative_input:
                    if isinstance(col_arr, np.ndarray) and col_arr.ndim >= 1:
                        tiled_input.append(np.repeat(col_arr, K, axis=0))
                    else:
                        tiled_input.append(col_arr)
                tiled_input = list(tiled_input)
                flat_cand = cand_items.reshape(-1)
                tiled_input[self._item_input_index] = flat_cand
                tiled_input = self._apply_candidate_features(
                    tuple(tiled_input),
                    flat_cand,
                )

                import torch as _torch

                with _torch.no_grad():
                    inputs = [
                        _torch.as_tensor(t).to(self._device) if isinstance(t, np.ndarray) else t for t in tiled_input
                    ]
                    s = self._model(*inputs)
                    if s.dim() > 1:
                        s = s.squeeze(-1)
                    scores = s.detach().cpu().numpy().reshape(num_of_negatives, K)

                argmax = scores.argmax(axis=1)
                new_items = cand_items[np.arange(num_of_negatives), argmax]

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = new_items
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(np.concatenate([p, n_]) for p, n_ in zip(positive_input, negative_input))
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output


# =============================================================================
# Strategy 8b: DNS over a CandidateProvider pool (Phase 1.5 — train/inference
# distribution match: draw negatives from the same filter the submission uses)
# =============================================================================


class InteractionsWithDynamicNegativeMiningCandidateProviderDataset(
    InteractionsWithDynamicNegativeMiningDataset,
):
    """DNS hard-mining over candidates drawn from a CandidateProvider pool.

    Identical to ``InteractionsWithDynamicNegativeMiningDataset`` except the
    candidate source is the row-conditional eligible pool returned by
    ``project_config.candidate_provider_factory(provider_extras)``, instead
    of uniform-from-catalog. The model still picks the highest-scoring of K
    candidates per slot (DNS behaviour) — but now the K candidates come
    from the same distribution the submission filter will use at inference
    time.

    Use case: close the train-inference distribution gap when inference
    scores only a filtered candidate pool. If the inference filter is
    e.g. ``bm25 ∪ cf_neighbors`` then training on the same pool teaches the
    ranker to discriminate inside that distribution rather than treating
    BM25-eligible candidates as out-of-distribution noise.

    Performance note: the eligibility per row is **precomputed once** at
    ``__init__`` time and cached. Calling ``provider.get_candidates`` per
    sample inside multi-worker DataLoaders with BM25 (which has internal
    threading) deadlocks under PyTorch's ``fork`` start method — see
    Phase 1.5 incident notes. With ``precompute_eligibles=True`` (default),
    the provider is hit ~N_rows times sequentially in __init__ (~minutes
    for 80 k rows), then sampling becomes O(K) per slot from a numpy
    array; safe for any DataLoader worker count.

    Args:
        provider_extras: forwarded to ``resolve_candidate_provider`` (e.g.
            ``{"strategy": "<project-defined strategy>", "min_pool": 500}``).
        precompute_eligibles: if True (default), call provider once per
            row at init and cache the result. Set False to use the legacy
            per-row provider call (slower; deadlocks under multi-worker
            BM25 — only safe with num_workers=0).
        eligibles_cap: optional max items kept per row when precomputing.
            ``None`` (default) keeps the full eligible set; an int caps it
            via random subsample. Useful to bound memory when the pool is
            very wide (e.g. > 10 k items per row).

    Other args inherit from the parent (``candidates_per_slot``,
    ``warmup_steps``, ``negative_proportion``).
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        candidates_per_slot: int = 8,
        warmup_steps: int = 0,
        provider_extras: Optional[Dict[str, Any]] = None,
        precompute_eligibles: bool = True,
        eligibles_cap: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            data_frame=data_frame,
            embeddings_for_metadata=embeddings_for_metadata,
            project_config=project_config,
            index_mapping=index_mapping,
            negative_proportion=negative_proportion,
            candidates_per_slot=candidates_per_slot,
            warmup_steps=warmup_steps,
            *args,
            **kwargs,
        )

        from mars_rec.filtering import resolve_candidate_provider

        self._provider = resolve_candidate_provider(
            project_config,
            index_mapping,
            provider_extras or {},
        )
        if self._provider is None:
            raise ValueError(
                "InteractionsWithDynamicNegativeMiningCandidateProviderDataset "
                "requires the project_config to declare a "
                "candidate_provider_factory."
            )

        item_map = index_mapping.get(project_config.item_column.name, {}) or {}
        self._full_catalog = real_indices(item_map)
        # Reset-index'd copy so iloc[src_pos_idx] is well-defined regardless
        # of any upstream reindexing / shuffling done by the prep pipeline.
        self._df_for_rows = data_frame.reset_index(drop=True)
        self._eligibles_cap = None if eligibles_cap is None else int(eligibles_cap)

        # Precompute per-row eligibles once. This avoids the deadlock that
        # was observed on Vertex when bm25s.retrieve was called per row
        # inside fork-spawned DataLoader workers (Phase 1.5 first wave).
        self._row_eligibles: Optional[List[np.ndarray]] = None
        if precompute_eligibles:
            self._row_eligibles = self._precompute_row_eligibles()
            # IMPORTANT: drop the provider AFTER precompute. The bm25s
            # retriever and the pre-computed embedding matrices inside
            # ``EmbeddingRetrievalFilter`` are GB-scale objects with
            # internal threading; leaving them attached causes
            # ``DataLoader`` worker spawn to hang for minutes (Phase 1.5
            # second wave hang). With the eligibles cached in
            # ``self._row_eligibles``, the provider is no longer needed
            # at sample time — drop it so each worker has a small
            # picklable / fork-friendly dataset.
            self._provider = None
            self._df_for_rows = None  # also drop the source DataFrame
            self._full_catalog = None

    def _precompute_row_eligibles(self) -> List[np.ndarray]:
        """Compute and cache provider eligibles for every row.

        Runs single-threaded in __init__ — the per-row provider cost (~3 ms
        per BM25 retrieve) is paid once, not per training step. Forked
        DataLoader workers inherit the cached list via copy-on-write, so
        memory is not duplicated.
        """
        import logging as _logging
        import time as _time

        _logger = _logging.getLogger(__name__)

        n = len(self._df_for_rows)
        result: List[np.ndarray] = [None] * n  # type: ignore[list-item]
        _logger.info(
            "Precomputing CandidateProvider eligibles for %d rows ...",
            n,
        )
        t0 = _time.time()
        log_every = max(1, n // 20)  # ~20 progress lines
        for i in range(n):
            row = self._df_for_rows.iloc[i].to_dict()
            elig = np.asarray(
                self._provider.get_candidates(row, self._full_catalog),
                dtype=np.int32,
            )
            if self._eligibles_cap is not None and elig.size > self._eligibles_cap:
                elig = np.random.choice(
                    elig,
                    size=self._eligibles_cap,
                    replace=False,
                ).astype(np.int32, copy=False)
            result[i] = elig
            if (i + 1) % log_every == 0 or (i + 1) == n:
                _logger.info(
                    "  %d/%d rows  elapsed=%.1fs",
                    i + 1,
                    n,
                    _time.time() - t0,
                )
        _logger.info(
            "CandidateProvider precomputation done in %.1fs (%d rows)",
            _time.time() - t0,
            n,
        )
        return result

    def _draw_candidates(
        self,
        sample_positive_indices: List[int],
        original_items: np.ndarray,
        K: int,
    ) -> np.ndarray:
        """Draw K candidates per slot from each row's CandidateProvider pool.

        Uses the precomputed ``self._row_eligibles`` (no provider calls at
        sample time) when available. When precomputation is disabled,
        falls back to the legacy per-row provider call — only safe with
        ``num_workers=0`` because of fork deadlocks with BM25 threading.

        Falls back to uniform-from-catalog when the row's eligible pool is
        empty (e.g. cold-start row with empty user_query_text).
        """
        num_negatives = len(original_items)
        cand_items = np.empty((num_negatives, K), dtype=np.int64)
        use_cache = self._row_eligibles is not None
        for i, src_pos_idx in enumerate(sample_positive_indices):
            exclude = int(original_items[i])
            if use_cache:
                eligible = self._row_eligibles[int(src_pos_idx)]
            else:
                try:
                    row = self._df_for_rows.iloc[int(src_pos_idx)].to_dict()
                except (IndexError, KeyError):
                    row = {}
                eligible = np.asarray(
                    self._provider.get_candidates(row, self._full_catalog),
                    dtype=np.int64,
                )
            # Exclude the positive item from negatives.
            if eligible.size > 0:
                eligible = eligible[eligible != exclude]
            if eligible.size == 0:
                # No eligibles — fall back to uniform.
                for k in range(K):
                    cand_items[i, k] = self._random_negative(exclude)
                continue
            if eligible.size >= K:
                cand_items[i, :] = np.random.choice(
                    eligible,
                    size=K,
                    replace=False,
                )
            else:
                # Fewer than K eligibles — sample with replacement so we
                # always return a [K]-sized vector.
                cand_items[i, :] = np.random.choice(
                    eligible,
                    size=K,
                    replace=True,
                )
        return cand_items


class InteractionsWithMixedDnsUniformDataset(
    InteractionsWithDynamicNegativeMiningDataset,
):
    """Hybrid sampler: per-slot Bernoulli mix of uniform random + DNS hard mining.

    For each negative slot, with probability ``uniform_fraction`` draw a
    uniform random item; otherwise run standard DNS (sample K candidates,
    keep the highest-scoring under the current model). Default 0.3 / 0.7.

    Use case: blend DNS hardness with uniform-in-catalog diversity. Pure DNS
    can over-concentrate negatives in the model's confused region, hurting
    coverage; pure uniform misses the gradient signal of hard negatives.

    Args:
        uniform_fraction: ∈ [0, 1]. Probability a slot is filled via uniform
            random instead of DNS K-candidate mining. Default 0.3.

    Other args inherit from ``InteractionsWithDynamicNegativeMiningDataset``.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        candidates_per_slot: int = 8,
        warmup_steps: int = 0,
        uniform_fraction: float = 0.3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            data_frame=data_frame,
            embeddings_for_metadata=embeddings_for_metadata,
            project_config=project_config,
            index_mapping=index_mapping,
            negative_proportion=negative_proportion,
            candidates_per_slot=candidates_per_slot,
            warmup_steps=warmup_steps,
            *args,
            **kwargs,
        )
        if not (0.0 <= uniform_fraction <= 1.0):
            raise ValueError(f"uniform_fraction must be in [0, 1], got {uniform_fraction}")
        self._uniform_fraction = float(uniform_fraction)

    def __getitem__(self, indices: Union[int, List[int], slice]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        # Skip our parent's __getitem__ — we re-implement the negative-handling
        # block to add the per-slot uniform/DNS Bernoulli. Bypass directly to
        # InteractionsDataset.
        n = InteractionsDataset.__len__(self)
        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        positive_input, positive_output = InteractionsDataset.__getitem__(
            self,
            positive_indices,
        )

        if num_of_negatives > 0:
            sample_positive_indices = list(np.random.randint(0, n, size=num_of_negatives))
            negative_input, _ = InteractionsDataset.__getitem__(
                self,
                sample_positive_indices,
            )
            negative_output = self._convert_dtype(np.zeros(num_of_negatives), self._project_config.output_column.type)

            original_items = negative_input[self._item_input_index]
            self._step_counter += 1
            use_dns_phase = self._model is not None and self._step_counter > self._warmup_steps

            if not use_dns_phase or self._uniform_fraction >= 1.0:
                # Warmup OR all-uniform: 1 candidate per slot, no scoring.
                draws = self._draw_candidates(
                    sample_positive_indices,
                    original_items,
                    K=1,
                )
                new_items = draws[:, 0]
            else:
                # Per-slot Bernoulli on which slots use DNS vs uniform.
                # Uniform slots: draw 1 candidate each (no scoring).
                # DNS slots: draw K candidates and keep argmax(score).
                uniform_mask = np.random.random(num_of_negatives) < self._uniform_fraction
                dns_mask = ~uniform_mask
                new_items = np.empty(num_of_negatives, dtype=np.int64)

                # Uniform slots — 1 candidate each.
                if uniform_mask.any():
                    unif_idx = np.flatnonzero(uniform_mask)
                    unif_pos_idxs = [sample_positive_indices[i] for i in unif_idx]
                    unif_originals = original_items[unif_idx]
                    unif_draws = self._draw_candidates(
                        unif_pos_idxs,
                        unif_originals,
                        K=1,
                    )
                    new_items[unif_idx] = unif_draws[:, 0]

                # DNS slots — K candidates, score, argmax.
                if dns_mask.any():
                    dns_idx = np.flatnonzero(dns_mask)
                    dns_pos_idxs = [sample_positive_indices[i] for i in dns_idx]
                    dns_originals = original_items[dns_idx]
                    K = self._candidates_per_slot
                    cand_items = self._draw_candidates(
                        dns_pos_idxs,
                        dns_originals,
                        K=K,
                    )
                    # Build the input tile only for DNS slots.
                    dns_negative_input = []
                    for col_arr in negative_input:
                        if isinstance(col_arr, np.ndarray) and col_arr.ndim >= 1:
                            dns_negative_input.append(col_arr[dns_idx])
                        else:
                            dns_negative_input.append(col_arr)
                    tiled_input = []
                    for col_arr in dns_negative_input:
                        if isinstance(col_arr, np.ndarray) and col_arr.ndim >= 1:
                            tiled_input.append(np.repeat(col_arr, K, axis=0))
                        else:
                            tiled_input.append(col_arr)
                    flat_cand = cand_items.reshape(-1)
                    tiled_input[self._item_input_index] = flat_cand
                    tiled_input = self._apply_candidate_features(
                        tuple(tiled_input),
                        flat_cand,
                    )

                    import torch as _torch

                    with _torch.no_grad():
                        inputs = [
                            _torch.as_tensor(t).to(self._device) if isinstance(t, np.ndarray) else t
                            for t in tiled_input
                        ]
                        s = self._model(*inputs)
                        if s.dim() > 1:
                            s = s.squeeze(-1)
                        scores = s.detach().cpu().numpy().reshape(len(dns_idx), K)
                    argmax = scores.argmax(axis=1)
                    new_items[dns_idx] = cand_items[
                        np.arange(len(dns_idx)),
                        argmax,
                    ]

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = new_items
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(np.concatenate([p, n_]) for p, n_ in zip(positive_input, negative_input))
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output


# =============================================================================
# Strategy 9: Mixed Hard Negative Sampling (ANN + popularity + uniform)
# =============================================================================


class InteractionsWithMixedHardNegativeDataset(InteractionsDataset):
    """Mixture of three negative sources controlled by per-class proportions.

    For each negative slot, multinomially sample a source ∈
    {ann, popularity, uniform} according to the configured ratios, then draw
    from the chosen distribution. ANN/popularity fall back to uniform when
    the source is unavailable for the row's positive item.

    Defaults: 50% ANN-hard / 30% popularity / 20% uniform.

    Args:
        neighbors_path: path to the pickle produced by build_cf_bpr_neighbors.
        ann_ratio: fraction of negatives drawn from ANN neighbours.
        popularity_ratio: fraction drawn from popularity-weighted distribution.
        uniform_ratio: fraction drawn uniformly. Sum is renormalised.
        smoothing_factor: exponent for popularity (sqrt = 0.5 default).
        popularity_column: optional external popularity column.
        top_k_use: cap on neighbours considered per item.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        embeddings_for_metadata: Optional[Dict[str, np.ndarray]],
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        negative_proportion: float = 0.8,
        neighbors_path: Optional[str] = None,
        ann_ratio: float = 0.5,
        popularity_ratio: float = 0.3,
        uniform_ratio: float = 0.2,
        smoothing_factor: float = 0.5,
        popularity_column: Optional[str] = None,
        top_k_use: int = 50,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(data_frame, embeddings_for_metadata, project_config, index_mapping, *args, **kwargs)
        self._negative_proportion = negative_proportion
        self._max_item_idx = int(data_frame[project_config.item_column.name].max())

        total = ann_ratio + popularity_ratio + uniform_ratio
        if total <= 0:
            raise ValueError("Sum of mixture ratios must be > 0.")
        self._ann_ratio = ann_ratio / total
        self._pop_ratio = popularity_ratio / total
        self._unif_ratio = uniform_ratio / total

        self._neighbors_by_idx: List[np.ndarray] = []
        if self._ann_ratio > 0.0:
            if not neighbors_path or not os.path.isfile(neighbors_path):
                raise FileNotFoundError(f"Mixed sampler with ann_ratio>0 needs neighbors_path; got {neighbors_path!r}")
            import pickle as _pickle

            with open(neighbors_path, "rb") as f:
                uuid_to_neighbors: Dict[Any, List[Any]] = _pickle.load(f)

            item_idx_map = index_mapping.get(project_config.item_column.name, {}) or {}
            self._neighbors_by_idx = [np.empty(0, dtype=np.int64) for _ in range(self._max_item_idx + 1)]
            for uuid, neigh_uuids in uuid_to_neighbors.items():
                if uuid not in item_idx_map:
                    continue
                src_idx = int(item_idx_map[uuid])
                if not (0 <= src_idx <= self._max_item_idx):
                    continue
                neigh_idx = [
                    int(item_idx_map[u])
                    for u in neigh_uuids[:top_k_use]
                    if u in item_idx_map and int(item_idx_map[u]) > 1
                ]
                if neigh_idx:
                    self._neighbors_by_idx[src_idx] = np.asarray(neigh_idx, dtype=np.int64)

        freq = np.zeros(self._max_item_idx + 1, dtype=np.float64)
        if popularity_column and popularity_column in data_frame.columns:
            item_col = project_config.item_column.name
            item_pop = data_frame.groupby(item_col)[popularity_column].first()
            for item_idx, pop_score in item_pop.items():
                if 0 <= int(item_idx) <= self._max_item_idx:
                    freq[int(item_idx)] = max(0.0, float(pop_score))
        else:
            counts = data_frame[project_config.item_column.name].value_counts()
            for item_idx, count in counts.items():
                if 0 <= int(item_idx) <= self._max_item_idx:
                    freq[int(item_idx)] = float(count)
        self._popularity_probs = self._restrict_to_pool(np.power(freq, smoothing_factor))

    def __len__(self) -> int:
        return super().__len__() + int((1 / (1 - self._negative_proportion) - 1) * super().__len__())

    def _sample_one(self, src: int, source: str) -> int:
        if source == "ann":
            if self._neighbors_by_idx and 0 <= src <= self._max_item_idx and self._neighbors_by_idx[src].size > 0:
                pick = int(np.random.choice(self._neighbors_by_idx[src]))
                return pick if pick != src else self._random_negative(src)
            source = "pop"
        if source == "pop":
            pick = int(
                np.random.choice(
                    len(self._popularity_probs),
                    p=self._popularity_probs,
                )
            )
            return pick if pick != src else self._random_negative(src)
        return self._random_negative(src)

    def __getitem__(self, indices: Union[int, List[int], slice]) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, slice):
            indices = list(range(len(self))[indices])

        n = super().__len__()
        positive_indices = [index for index in indices if index < n]
        num_of_negatives = len(indices) - len(positive_indices)
        positive_input, positive_output = super().__getitem__(positive_indices)

        if num_of_negatives > 0:
            sample_positive_indices = list(np.random.randint(0, n, size=num_of_negatives))
            negative_input, _ = super().__getitem__(sample_positive_indices)
            negative_output = self._convert_dtype(np.zeros(num_of_negatives), self._project_config.output_column.type)

            original_items = negative_input[self._item_input_index]

            sources = np.random.choice(
                ["ann", "pop", "unif"],
                size=num_of_negatives,
                p=[self._ann_ratio, self._pop_ratio, self._unif_ratio],
            )
            new_items = np.array(
                [self._sample_one(int(src), source=sources[i]) for i, src in enumerate(original_items)], dtype=np.int64
            )

            negative_input = list(negative_input)
            negative_input[self._item_input_index] = new_items
            negative_input = self._apply_candidate_features(
                negative_input,
                negative_input[self._item_input_index],
            )
            negative_input = tuple(negative_input)

            if positive_indices:
                input_ = tuple(np.concatenate([p, n_]) for p, n_ in zip(positive_input, negative_input))
                output = _concat_outputs(positive_output, negative_output)
            else:
                input_ = negative_input
                output = negative_output
        else:
            input_ = positive_input
            output = positive_output

        return input_, output
