"""Categorical-to-integer mappings used for embedding lookup.

The framework keeps user / item / categorical columns as strings on
disk and indexes them through a per-column ``{raw_value: int_id}``
dict at training time. Three integer slots are reserved at the head
of every mapping:

- ``0`` — unknown (``defaultdict`` fallback for never-seen values),
- ``1`` — ``None`` / ``NaN``,
- ``2`` — ``-1`` (and the string ``"-1"``), used as a generic pad
  token for sequence columns.

Real ids start at ``3``. The reserved slots are why model code
constructs ``nn.Embedding`` with ``num_embeddings = n_items + 3``
(or just uses ``max(mapping.values()) + 1`` as :class:`RecommenderModule`
does internally).
"""

import functools
import math
from collections import defaultdict
from typing import Dict, Any, List, Iterable, Mapping

import numpy as np
import pandas as pd

from mars_rec.meta_config import ProjectConfig, IOType
from multiprocessing import Pool
import os


def create_index_mapping(
    indexable_values: Iterable, include_unknown: bool = True, include_none: bool = True
) -> Dict[Any, int]:
    """Build a ``{value: int_index}`` lookup from an iterable of raw values.

    Sorts the deduplicated string-cast values and assigns sequential
    ids. When ``include_unknown=True`` (the default) returns a
    ``defaultdict(int)`` so missing keys silently map to ``0``;
    otherwise returns a plain dict (raises ``KeyError`` on misses).
    The ``include_none`` flag prepends the ``None`` / ``-1`` sentinel
    slots (1 and 2) used by sequence padding.
    """
    indexable_values = list(sorted(set(str(value) for value in indexable_values if not pd.isnull(value))))
    if include_none:
        indexable_values = [None, -1] + indexable_values

    if include_unknown:
        indices = np.arange(1, len(indexable_values) + 1)
        mapping = defaultdict(int, zip(indexable_values, indices))  # Unkown = 0
    else:
        indices = np.arange(0, len(indexable_values))
        mapping = dict(zip(indexable_values, indices))

    if include_none:
        mapping[np.nan] = 1  # Both None and nan are treated the same
        mapping["-1"] = 2  # Both pad -1 and "-1"

    return mapping


def is_reserved_key(raw_val: Any) -> bool:
    """True for the sentinel keys of :func:`create_index_mapping` (None, NaN, -1)."""
    if raw_val is None:
        return True
    if isinstance(raw_val, float) and math.isnan(raw_val):
        return True
    return str(raw_val) == "-1"


def real_indices(mapping: Mapping[Any, int]) -> np.ndarray:
    """Sorted ``int64`` indices of the real values in ``mapping``.

    Drops the unknown slot (``0``, which a ``defaultdict`` also hands to
    any key looked up after construction) and the sentinel keys
    (:func:`is_reserved_key`). This is the single definition of "a real
    catalog item" shared by training, negative sampling and inference.
    """
    idx = {int(v) for k, v in mapping.items() if int(v) != 0 and not is_reserved_key(k)}
    return np.asarray(sorted(idx), dtype=np.int64)


def create_index_mapping_from_arrays(
    indexable_arrays: Iterable[list],
    include_unknown: bool = True,
    include_none: bool = True,
) -> Dict[Any, int]:
    """Like :func:`create_index_mapping` but flattens a per-row iterable of lists.

    Use this for ``IOType.INDEXABLE_ARRAY`` columns (sequences of ids
    per row) where the vocabulary is the union of every element
    across rows.
    """
    all_values = set(str(value) for values in indexable_arrays for value in values)
    return create_index_mapping(all_values, include_unknown, include_none)


def map_array(values: list, mapping: dict) -> List[int]:
    """Apply ``mapping`` (a ``{str: int}`` dict) to each entry in ``values``."""
    return [int(mapping[str(value)]) for value in values]


def transform_with_indexing(
    df: pd.DataFrame,
    index_mapping: Dict[str, dict],
    project_config: ProjectConfig,
):
    """Rewrite every ``INDEXABLE`` / ``INDEXABLE_ARRAY`` column to its int ids.

    Mutates ``df`` in place. Mapped ``INDEXABLE`` columns become
    ``int`` dtype; ``INDEXABLE_ARRAY`` columns are remapped in
    parallel via ``multiprocessing.Pool`` (chunksize tuned by CPU
    count). Columns not present in ``project_config.all_columns`` are
    left untouched.
    """
    if df is None:
        return None

    # Pre-compute column lookups to avoid repeated get_column_by_name calls
    column_lookup = {}
    array_columns = []
    for key in index_mapping:
        column = project_config.get_column_by_name(key)
        if column and key in df:
            column_lookup[key] = column
            if column.type == IOType.INDEXABLE_ARRAY:
                array_columns.append(key)

    if array_columns:
        chunksize = max(1, len(df) // (os.cpu_count() * 4))
        with Pool(os.cpu_count()) as pool:
            for key, mapping in index_mapping.items():
                column = column_lookup.get(key)
                if column is None:
                    continue
                if column.type == IOType.INDEXABLE:
                    col = df[key]
                    if col.dtype != object:
                        col = col.astype(str)
                    df[key] = col.map(mapping).astype(int)
                elif column.type == IOType.INDEXABLE_ARRAY:
                    df[key] = pool.map(
                        functools.partial(map_array, mapping=mapping),
                        df[key],
                        chunksize=chunksize,
                    )
    else:
        for key, mapping in index_mapping.items():
            column = column_lookup.get(key)
            if column is None:
                continue
            if column.type == IOType.INDEXABLE:
                col = df[key]
                if col.dtype != object:
                    col = col.astype(str)
                df[key] = col.map(mapping).astype(int)
