"""Grab-bag of small utilities used across the framework.

Highlights:

- :func:`parallel_literal_eval` — parsing of CSV cells holding Python
  literals (used by the Dataset preprocessing for array columns);
  multi-process only for large inputs (see :func:`use_parallel`).
- :func:`reduce_df_mem` — narrow numeric columns to the smallest dtype
  that fits, useful for big interaction frames.
- :class:`JsonEncoder` — ``json.JSONEncoder`` subclass that handles
  numpy scalars and arrays so ``params.json`` / ``metadata.json``
  round-trip cleanly.
- Miscellaneous helpers (filename sanitisation, confidence intervals,
  date conversion, file walking, …).
"""

import os
import ast
import string
import unicodedata
import logging
from datetime import datetime, timedelta
from multiprocessing.pool import Pool
from typing import Union
import json
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import randrange


logger = logging.getLogger(__name__)

# Filename sanitisation adapted from
# https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8

valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
char_limit = 255


def random_date(start, l):
    """Yield ``l`` random ``datetime`` values within an hour of ``start``."""
    current = start
    while l > 0:
        curr = current + timedelta(minutes=randrange(60))
        yield curr
        l -= 1


def clean_filename(filename, whitelist=valid_filename_chars, replace=" "):
    """Sanitise ``filename`` for safe filesystem use.

    Replaces every character in ``replace`` with an underscore, strips
    non-ASCII characters via NFKD normalisation, then keeps only
    characters from ``whitelist``. Truncates to 255 chars and logs a
    warning when truncation happens (collisions are possible after).
    """
    # replace spaces
    for r in replace:
        filename = filename.replace(r, "_")

    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize("NFKD", filename).encode("ASCII", "ignore").decode()

    # keep only whitelisted chars
    cleaned_filename = "".join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename) > char_limit:
        logger.warning(
            "Filename truncated to %d chars — collisions are possible",
            char_limit,
        )
    return cleaned_filename[:char_limit]


def chunks(l: Union[list, range], n: int) -> Union[list, range]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


PARALLEL_MIN_ROWS_ENV = "MARS_REC_PARALLEL_MIN_ROWS"
"""Environment variable overriding :data:`DEFAULT_PARALLEL_MIN_ROWS`."""

DEFAULT_PARALLEL_MIN_ROWS = 2_000_000
"""Row count from which per-row parsing / remapping switches to a ``multiprocessing.Pool``.

Starting a pool is not free: every ``spawn`` worker re-imports the framework
(torch, Lightning, pandas), which costs several seconds per pool on macOS. The
work it parallelises (``json.loads`` of short lists, dict lookups) runs at
~1M rows/s in a single process, so below a few million rows the serial path
is faster.
"""


def parallel_min_rows() -> int:
    """Row threshold for :func:`use_parallel` (``MARS_REC_PARALLEL_MIN_ROWS``, default 2M).

    ``0`` forces the pool for every non-empty input; a negative value disables it.
    """
    raw = os.environ.get(PARALLEL_MIN_ROWS_ENV, "").strip()
    if not raw:
        return DEFAULT_PARALLEL_MIN_ROWS
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Ignoring %s=%r (not an integer); using %d", PARALLEL_MIN_ROWS_ENV, raw, DEFAULT_PARALLEL_MIN_ROWS
        )
        return DEFAULT_PARALLEL_MIN_ROWS


def use_parallel(n_rows: int) -> bool:
    """Whether ``n_rows`` rows are worth a ``multiprocessing.Pool`` (see :func:`parallel_min_rows`)."""
    threshold = parallel_min_rows()
    return threshold >= 0 and n_rows > 0 and n_rows >= threshold


def parallel_literal_eval(series: Union[pd.Series, np.ndarray], pool: Pool = None, use_tqdm: bool = True) -> list:
    """Parse a Series of Python-literal strings, in parallel for large inputs.

    Skips parsing when the first element is already a parsed object
    (lets callers run the function unconditionally on a column whose
    values may have been deserialised on a previous pass). Uses ``pool``
    when given; otherwise spins up a one-shot ``multiprocessing.Pool``
    only when :func:`use_parallel` says the input is large enough, and
    parses in-process below that.
    """
    if len(series) == 0:
        return []

    # Skip parsing if elements are already parsed (not strings)
    first = series.iloc[0] if isinstance(series, pd.Series) else series[0]
    if not isinstance(first, str):
        return series.tolist() if hasattr(series, "tolist") else list(series)

    if pool:
        return _parallel_literal_eval(series, pool, use_tqdm)
    if use_parallel(len(series)):
        with Pool(os.cpu_count()) as p:
            return _parallel_literal_eval(series, p, use_tqdm)
    return [literal_eval_if_str(value) for value in series]


def literal_eval_if_str(element):
    """Parse ``element`` as JSON if it is a string, else return it unchanged.

    Falls back to :func:`ast.literal_eval` when ``json.loads`` rejects
    the string — handles single-quoted Python list literals which JSON
    refuses but pandas / pickle round-trips produce.
    """
    if isinstance(element, str):
        try:
            return json.loads(element)
        except (json.JSONDecodeError, ValueError):
            return ast.literal_eval(element)
    return element


def _pad_sequence(seq, pad) -> np.ndarray:
    if seq is None:
        return None
    else:
        return (([0] * pad) + seq)[-pad:]


def to_array(xs):
    """Coerce ``xs`` (string literal of ints or list of pairs) to a flat list of ints.

    Legacy helper used by older project pipelines to normalise mixed
    string / nested list columns. Returns ``None`` when ``xs`` is ``None``.
    """
    return (
        [
            literal_eval_if_str(c) if isinstance(literal_eval_if_str(c), int) else literal_eval_if_str(c)[0]
            for c in literal_eval_if_str(xs)
        ]
        if xs is not None
        else None
    )


def _parallel_literal_eval(series: Union[pd.Series, np.ndarray], pool: Pool, use_tqdm: bool = True) -> list:
    chunksize = max(1, len(series) // (os.cpu_count() * 4))
    if use_tqdm:
        return list(tqdm(pool.map(literal_eval_if_str, series, chunksize=chunksize), total=len(series)))
    else:
        return pool.map(literal_eval_if_str, series, chunksize=chunksize)


def date_to_day_of_week(date: str) -> int:
    """Return the day-of-week (0=Sunday … 6=Saturday) for an ISO date string."""
    return int(datetime.strptime(date, "%Y-%m-%d").strftime("%w"))


def date_to_day_of_month(date: str) -> int:
    """Return the day-of-month (1..31) for an ISO date string."""
    return int(datetime.strptime(date, "%Y-%m-%d").strftime("%d"))


def get_all_file_paths(directory):
    """Walk ``directory`` and return the full path of every file beneath it."""
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths


def mean_confidence_interval(data, confidence=0.95):
    """Return ``(mean, half_width)`` of the confidence interval for ``data``.

    Strips ``NaN`` values, then uses Student's t to compute the
    margin of error at the requested confidence level (default 95 %).
    """
    data = np.array(data)
    data = data[~np.isnan(data)]
    a = 1.0 * data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


class JsonEncoder(json.JSONEncoder):
    """``json.JSONEncoder`` that natively handles numpy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


def reduce_df_mem(df, without_columns=[]):
    """Downcast numeric columns to the smallest dtype that fits their range.

    Iterates over every column not in ``without_columns`` and narrows
    ``int`` / ``float`` columns by inspecting their min / max. Useful
    when loading multi-million-row interaction frames before they are
    handed to a Dataset. Logs the before/after memory at DEBUG level.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if col in without_columns:
            continue
        if col_type != object:  # noqa: E721  (dtype vs. object comparison is intentional)
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            elif str(col_type)[:5] == "float":
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if start_mem > 0:
        logger.debug(
            "reduce_df_mem: %.1f → %.1f MiB (-%.1f%%)",
            start_mem,
            end_mem,
            100 * (start_mem - end_mem) / start_mem,
        )

    return df
