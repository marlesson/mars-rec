"""Per-candidate feature recomputation, shared by training, evaluation and inference.

When a row is reused with a different item id (a negative sample at train
time, or one of the ranking candidates at evaluation / inference time), any
input column that describes *the item* must describe the new item, not the
original one. Otherwise the model learns to spot the mismatch instead of
learning the ranking.

Projects opt in through two entries of ``ProjectConfig.dataset_extra_params``:

``candidate_feature_lookup``
    A dict ``{name: values_by_item_idx}``, or a 0-/1-arg callable
    ``() -> dict`` / ``(index_mapping) -> dict`` that builds it lazily.
    Every key that is also an input column (other than the item column) is
    replaced by ``values[candidate_idx]``: a **direct lookup**. Keys that
    start with ``_`` are private data for the hook below and never touch a
    column.

``candidate_feature_fn``
    Optional callable ``(columns, candidate_idx, lookup) -> dict`` for
    features that must be *computed* from the candidate and the row (for
    example "how many times does the candidate's category appear in the
    row's history"). ``columns`` maps each input column name to its batch
    array (direct lookups already applied); the return value maps column
    names to new batch arrays. Values are cast to the column's declared
    :class:`~mars_rec.meta_config.IOType`.

The same :func:`apply_candidate_features` runs in the negative samplers, the
training-time evaluation and :func:`mars_rec.inference.score_candidates_bce`,
so the three paths cannot drift apart.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from mars_rec.meta_config import Column, IOType

LOOKUP_PARAM = "candidate_feature_lookup"
FN_PARAM = "candidate_feature_fn"

CandidateFeatureFn = Callable[[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]], Mapping[str, Any]]


def convert_dtype(value: Any, type: IOType) -> Any:
    """Cast a column's raw values to the dtype the model receives for ``type``."""
    if type == IOType.INDEXABLE:
        return np.asarray(value).astype(np.int64)
    if type == IOType.NUMBER:
        return np.asarray(value).astype(np.float32)
    if type in (IOType.INT_ARRAY, IOType.INDEXABLE_ARRAY, IOType.SEQUENCE):
        # Fast path when every entry has the same length (padded sequences);
        # falls back to a ragged object array otherwise.
        try:
            return np.asarray(list(value), dtype=np.int64)
        except (ValueError, TypeError):
            return _ragged(value, np.int64)
    if type == IOType.FLOAT_ARRAY:
        try:
            return np.asarray(list(value), dtype=np.float32)
        except (ValueError, TypeError):
            return _ragged(value, np.float32)
    return value


def _ragged(value: Any, dtype: type) -> np.ndarray:
    """1-D object array of per-row arrays (numpy refuses to infer ragged shapes)."""
    rows = list(value)
    out = np.empty(len(rows), dtype=object)
    for i, v in enumerate(rows):
        out[i] = np.asarray(v, dtype=dtype)
    return out


def _required_positional_arity(fn: Callable) -> int:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 0
    return len(
        [
            p
            for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect.Parameter.empty
        ]
    )


def resolve_candidate_lookup(raw: Any, index_mapping: Dict[str, Dict[Any, int]]) -> Optional[Dict[str, Any]]:
    """Materialise a ``candidate_feature_lookup`` value.

    :param raw: ``None``, a dict, or a 0-/1-arg callable returning a dict.
    :param index_mapping: Passed to a 1-arg callable.
    :returns: The lookup dict, or ``None`` when ``raw`` is ``None``.
    :raises TypeError: When ``raw`` (or what the callable returns) is not a dict.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if not callable(raw):
        raise TypeError(f"{LOOKUP_PARAM} must be a dict or callable, got {type(raw).__name__}")
    materialised = raw(index_mapping) if _required_positional_arity(raw) >= 1 else raw()
    if not isinstance(materialised, dict):
        raise TypeError(f"{LOOKUP_PARAM} callable must return a dict, got {type(materialised).__name__}")
    return materialised


def validate_candidate_feature_fn(fn: Any) -> Optional[CandidateFeatureFn]:
    """Return ``fn`` when it is ``None`` or callable; raise ``TypeError`` otherwise."""
    if fn is not None and not callable(fn):
        raise TypeError(f"{FN_PARAM} must be callable, got {type(fn).__name__}")
    return fn


_warned_private_keys = False


def _warn_private_keys_without_fn(lookup: Dict[str, Any]) -> None:
    """Warn once when the lookup carries hook-only data but no hook is configured.

    Before ``candidate_feature_fn`` existed the framework recomputed some
    derived columns implicitly from ``_``-prefixed lookup keys. Those columns
    are now left untouched unless the project registers the hook.
    """
    global _warned_private_keys
    if _warned_private_keys:
        return
    private = sorted(k for k in lookup if isinstance(k, str) and k.startswith("_"))
    if private:
        _warned_private_keys = True
        warnings.warn(
            f"{LOOKUP_PARAM} has private keys {private} but no {FN_PARAM} is configured. "
            "Derived candidate features are no longer recomputed by the framework; register a "
            f"{FN_PARAM} in dataset_extra_params (see docs/concepts/project-config.md).",
            UserWarning,
            stacklevel=3,
        )


def _direct_lookup(values: Any, candidate_idx: np.ndarray) -> Any:
    if isinstance(values, np.ndarray):
        return values[candidate_idx]
    # Lists of lists (e.g. a ragged multi-valued attribute per item).
    return [values[i] for i in candidate_idx]


def apply_candidate_features(
    columns: Dict[str, Any],
    candidate_idx: np.ndarray,
    input_columns: List[Column],
    lookup: Optional[Dict[str, Any]],
    fn: Optional[CandidateFeatureFn] = None,
    item_column_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the columns that change when each row's item becomes ``candidate_idx``.

    :param columns: ``{input_column_name: batch_values}`` for the rows being rewritten.
        Columns absent from the dict are ignored.
    :param candidate_idx: Item index of the candidate for each row, shape ``[B]``.
    :param input_columns: The project's input columns (for dtype casting).
    :param lookup: Resolved ``candidate_feature_lookup`` dict, or ``None``.
    :param fn: Optional ``candidate_feature_fn`` hook.
    :param item_column_name: The item column, which is never overwritten.
    :returns: ``{column_name: new_batch_values}`` cast to each column's IOType.
        Empty when neither ``lookup`` nor ``fn`` is configured.
    :raises KeyError: When ``fn`` returns a column that is not an input column.
    """
    if not lookup and fn is None:
        return {}
    lookup = lookup or {}
    col_types = {c.name: c.type for c in input_columns}
    cand = np.asarray(candidate_idx, dtype=np.int64)

    updates: Dict[str, Any] = {}
    for name, values in lookup.items():
        if not isinstance(name, str) or name.startswith("_") or name == item_column_name:
            continue
        if name in columns and name in col_types:
            updates[name] = convert_dtype(_direct_lookup(values, cand), col_types[name])

    if fn is None:
        _warn_private_keys_without_fn(lookup)
        return updates

    computed = fn({**columns, **updates}, cand, lookup) or {}
    for name, values in computed.items():
        if name not in col_types:
            raise KeyError(f"{FN_PARAM} returned {name!r}, which is not an input column")
        if name == item_column_name:
            raise KeyError(f"{FN_PARAM} must not overwrite the item column {name!r}")
        updates[name] = convert_dtype(values, col_types[name])
    return updates


def apply_candidate_features_to_frame(
    frame: pd.DataFrame,
    candidate_idx: np.ndarray,
    input_columns: List[Column],
    lookup: Optional[Dict[str, Any]],
    fn: Optional[CandidateFeatureFn] = None,
    item_column_name: Optional[str] = None,
) -> pd.DataFrame:
    """DataFrame flavour of :func:`apply_candidate_features`; rewrites ``frame`` in place and returns it."""
    if not lookup and fn is None:
        return frame
    columns: Dict[str, Any] = {}
    for c in input_columns:
        if c.name not in frame.columns:
            continue
        raw = frame[c.name].values
        try:
            columns[c.name] = convert_dtype(raw, c.type)
        except (ValueError, TypeError):
            columns[c.name] = raw  # not castable (e.g. raw strings): hand it over as is
    updates = apply_candidate_features(columns, candidate_idx, input_columns, lookup, fn, item_column_name)
    for name, values in updates.items():
        frame[name] = list(values) if getattr(values, "ndim", 1) > 1 else values
    return frame
