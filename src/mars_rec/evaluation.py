"""Offline top-k ranking evaluation, shared by every evaluation entry point.

Training (``SupervisedModelTraining`` after fit), ``mars-rec evaluate`` and
:func:`mars_rec.inference.run_evaluate` all go through this module, so the
protocol and the metric definitions cannot drift apart.

Protocol
--------

1. **Queries.** Only *positive* test rows are ground truth: rows whose
   ``output_column`` is ``> 0`` (a click, a label of 1, a positive rating).
   When the output column *is* the item column (next-item / CE models)
   every row is positive. Positive rows that share the same model context
   (user column plus every non-item input column) form one **query**, so a
   collaborative-filtering user with five held-out items is one query with
   five relevant items, while a sequential model gets one query per event
   (each row carries its own history). See :func:`build_ranking_queries`.
2. **Candidates.** Each query is scored against the whole catalog, unless
   the project restricts the pool (a per-row ``available_items`` slate or a
   ``CandidateProvider``). No negative sampling: sampled metrics can rank
   models differently from the exact ones (Krichene & Rendle, KDD 2020).
3. **Seen items.** Items the user already interacted with in train /
   validation are removed from the candidates. A held-out item that was
   also seen in training is removed from the relevant set (it can no longer
   be recommended); queries left without relevant items are dropped.
4. **Ties** are broken uniformly at random with a fixed seed, so a model
   cannot gain hits from the order in which candidates are listed.
5. **Aggregation.** Metrics are computed per query, averaged per user, then
   averaged over users, so heavy users do not dominate the result.

Metrics (``<name>@k``)
----------------------

Accuracy, with ``R`` the relevant set of a query and ``hits`` its top-k:

- ``hit_rate``  — 1 when at least one relevant item is in the top-k.
- ``precision`` — ``|hits| / k``.
- ``recall``    — ``|hits| / |R|``.
- ``ndcg``      — binary-relevance DCG over the ideal DCG of ``min(|R|, k)`` hits.
- ``mrr``       — reciprocal rank of the first relevant item.
- ``map``       — average precision truncated at k, normalised by ``min(|R|, k)``.

Beyond accuracy, over every query's top-k list:

- ``coverage``       — fraction of the catalog recommended at least once.
- ``gini``           — Gini index of how often each catalog item is
  recommended (0 = perfectly even exposure, 1 = one item gets everything).
- ``avg_popularity`` — mean training popularity (number of users with a
  positive interaction) of the recommended items (ARP).
- ``novelty``        — mean self-information ``-log2(pop / n_users)`` of the
  recommended items; higher means less popular, more novel items.

The two popularity metrics need training interactions and are omitted when
none are available. :func:`evaluate_rankings` also scores a
**most-popular baseline** on the same queries and candidates, which gives
the model's numbers a reference point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_KS: Tuple[int, ...] = (5, 10, 20)

ACCURACY_METRICS: Tuple[str, ...] = ("ndcg", "hit_rate", "precision", "recall", "mrr", "map")
POPULARITY_METRICS: Tuple[str, ...] = ("avg_popularity", "novelty")


def ranking_metric_names(ks: Iterable[int] = DEFAULT_KS, with_popularity: bool = True) -> List[str]:
    """Every key :func:`compute_ranking_metrics` returns for ``ks``."""
    names = list(ACCURACY_METRICS) + ["coverage", "gini"]
    if with_popularity:
        names += list(POPULARITY_METRICS)
    return [f"{name}@{k}" for k in sorted(set(ks)) for name in names]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


@dataclass
class RankingQueries:
    """Test rows grouped into ranking queries.

    :ivar positions: Positional index (``df.iloc``) of each query's context row.
    :ivar relevant: Distinct relevant item indices of each query.
    :ivar users: User index of each query (per-user averaging and seen-item lookup).
    :ivar n_rows: Rows in the evaluated frame.
    :ivar n_positive_rows: Rows that count as ground truth.
    """

    positions: np.ndarray
    relevant: List[np.ndarray]
    users: np.ndarray
    n_rows: int = 0
    n_positive_rows: int = 0

    def __len__(self) -> int:
        return len(self.positions)

    def subset(self, keep: np.ndarray) -> "RankingQueries":
        """Return the queries at ``keep`` (positions into this object)."""
        keep = np.asarray(keep, dtype=np.int64)
        return RankingQueries(
            positions=self.positions[keep],
            relevant=[self.relevant[i] for i in keep],
            users=self.users[keep],
            n_rows=self.n_rows,
            n_positive_rows=self.n_positive_rows,
        )

    def sample(self, n: Optional[int], seed: int = 0) -> "RankingQueries":
        """Return at most ``n`` queries drawn uniformly without replacement."""
        if not n or len(self) <= n:
            return self
        keep = np.sort(np.random.default_rng(seed).choice(len(self), size=n, replace=False))
        return self.subset(keep)


def _hashable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


def positive_mask(df: pd.DataFrame, item_column: str, output_column: str) -> np.ndarray:
    """Rows that count as ground truth: ``output > 0``, or every row for item-valued targets."""
    if output_column == item_column or output_column not in df.columns:
        return np.ones(len(df), dtype=bool)
    values = pd.to_numeric(df[output_column], errors="coerce").to_numpy(dtype=np.float64)
    return np.nan_to_num(values, nan=0.0) > 0


def build_ranking_queries(
    df: pd.DataFrame,
    user_column: str,
    item_column: str,
    output_column: str,
    context_columns: Sequence[str] = (),
) -> RankingQueries:
    """Group the positive rows of an *indexed* frame into ranking queries.

    Rows with equal values in ``user_column`` and every ``context_columns``
    get the same model input apart from the item, hence the same ranking:
    they form one query whose relevant set is the union of their items.

    :param df: Framework-indexed test frame (item / user columns are int indices).
    :param user_column: The user column (always part of the grouping key).
    :param item_column: Column whose value is the relevant item. For
        next-item models this is the output column.
    :param output_column: Target column, used to keep only positive rows.
    :param context_columns: Other columns that define the model context.
        Leave out columns that are rewritten per candidate (item features).
    """
    positive = positive_mask(df, item_column, output_column)
    pos_idx = np.flatnonzero(positive)
    if len(pos_idx) == 0:
        empty = np.empty(0, dtype=np.int64)
        return RankingQueries(empty, [], empty, n_rows=len(df), n_positive_rows=0)

    key_columns = [user_column] + [
        c for c in context_columns if c in df.columns and c not in (user_column, item_column)
    ]
    keys = pd.DataFrame(
        {
            str(i): (df[c].iloc[pos_idx].map(_hashable) if df[c].dtype == object else df[c].iloc[pos_idx])
            for i, c in enumerate(key_columns)
        }
    ).reset_index(drop=True)
    group_ids = keys.groupby(list(keys.columns), sort=False, dropna=False).ngroup().to_numpy()

    items = pd.to_numeric(df[item_column].iloc[pos_idx], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    users = pd.to_numeric(df[user_column].iloc[pos_idx], errors="coerce").fillna(0).to_numpy(dtype=np.int64)

    order = np.argsort(group_ids, kind="stable")
    sorted_groups = group_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    ends = np.r_[starts[1:], len(order)]

    # Keep queries in the order their first row appears in ``df``.
    first_rows = order[starts]
    query_order = np.argsort(first_rows, kind="stable")

    positions, relevant, query_users = [], [], []
    for q in query_order:
        members = order[starts[q] : ends[q]]
        positions.append(pos_idx[members[0]])
        relevant.append(np.unique(items[members]))
        query_users.append(users[members[0]])

    return RankingQueries(
        positions=np.asarray(positions, dtype=np.int64),
        relevant=relevant,
        users=np.asarray(query_users, dtype=np.int64),
        n_rows=len(df),
        n_positive_rows=len(pos_idx),
    )


# ---------------------------------------------------------------------------
# Interaction history (seen items, popularity)
# ---------------------------------------------------------------------------


@dataclass
class InteractionHistory:
    """What the model saw in training, used to exclude seen items and measure popularity.

    :ivar seen: ``{user_idx: sorted item indices}`` the user interacted with.
    :ivar popularity: ``popularity[item_idx]`` = users with a positive interaction.
    :ivar n_users: Users with at least one positive interaction.
    """

    seen: Dict[int, np.ndarray] = field(default_factory=dict)
    popularity: Optional[np.ndarray] = None
    n_users: int = 0

    @classmethod
    def from_interactions(
        cls,
        users: np.ndarray,
        items: np.ndarray,
        positive: np.ndarray,
        n_items: int,
    ) -> "InteractionHistory":
        """Build from parallel arrays of indexed ``(user, item)`` pairs.

        :param users: User index per interaction.
        :param items: Item index per interaction.
        :param positive: Whether each interaction is positive (feeds popularity).
        :param n_items: Size of the item index space (``max index + 1``).
        """
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        positive = np.asarray(positive, dtype=bool)

        pairs = pd.DataFrame({"u": users, "i": items}).drop_duplicates()
        seen = {int(u): np.sort(g.to_numpy(dtype=np.int64)) for u, g in pairs.groupby("u")["i"]}

        pos_pairs = pd.DataFrame({"u": users[positive], "i": items[positive]}).drop_duplicates()
        popularity = np.zeros(int(n_items), dtype=np.float64)
        counts = pos_pairs["i"].value_counts()
        in_range = counts.index.to_numpy() < n_items
        popularity[counts.index.to_numpy()[in_range]] = counts.to_numpy()[in_range]
        return cls(seen=seen, popularity=popularity, n_users=int(pos_pairs["u"].nunique()))


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_candidates(
    scores: np.ndarray,
    candidates: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-``k`` ``(items, scores)`` by descending score, ties broken at random.

    NaN scores rank last.
    """
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) == 0 or k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    keyed = np.where(np.isnan(scores), -np.inf, scores)
    order = np.lexsort((rng.random(len(candidates)), -keyed))[:k]
    return candidates[order], scores[order]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _gini(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0 or len(counts) == 0:
        return 0.0
    x = np.sort(counts)
    n = len(x)
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * x).sum() / (n * total))


def _user_mean(values: np.ndarray, users: Optional[np.ndarray]) -> float:
    if len(values) == 0:
        return 0.0
    if users is None:
        return float(values.mean())
    return float(pd.Series(values).groupby(users).mean().mean())


def compute_ranking_metrics(
    ranked: Sequence[np.ndarray],
    relevant: Sequence[np.ndarray],
    ks: Iterable[int] = DEFAULT_KS,
    users: Optional[np.ndarray] = None,
    n_catalog_items: Optional[int] = None,
    item_popularity: Optional[np.ndarray] = None,
    n_popularity_users: int = 0,
) -> Dict[str, float]:
    """Accuracy and beyond-accuracy metrics for top-k lists (see module docstring).

    :param ranked: Recommended item indices per query, best first (at least ``max(ks)`` long when possible).
    :param relevant: Relevant item indices per query.
    :param ks: Cutoffs.
    :param users: User per query. When given, metrics are averaged per user first.
    :param n_catalog_items: Catalog size for ``coverage`` / ``gini``; defaults to the distinct recommended items.
    :param item_popularity: ``popularity[item_idx]``; enables ``avg_popularity`` and ``novelty``.
    :param n_popularity_users: Normaliser for ``novelty`` (users behind ``item_popularity``).
    :returns: ``{"<metric>@<k>": value}``. Empty when there are no queries.
    """
    if len(ranked) != len(relevant):
        raise ValueError(f"ranked ({len(ranked)}) and relevant ({len(relevant)}) must have the same length")
    if len(ranked) == 0:
        return {}
    ks = sorted(set(int(k) for k in ks))
    max_k = ks[-1]
    n_queries = len(ranked)
    users = None if users is None else np.asarray(users)

    # hits[q, j] = 1 when the item at rank j+1 of query q is relevant.
    hits = np.zeros((n_queries, max_k), dtype=np.float64)
    n_relevant = np.zeros(n_queries, dtype=np.float64)
    top = np.full((n_queries, max_k), -1, dtype=np.int64)
    for q, (items, rel) in enumerate(zip(ranked, relevant)):
        items = np.asarray(items, dtype=np.int64)[:max_k]
        top[q, : len(items)] = items
        hits[q, : len(items)] = np.isin(items, rel)
        n_relevant[q] = len(rel)
    n_relevant = np.maximum(n_relevant, 1.0)

    positions = np.arange(1, max_k + 1, dtype=np.float64)
    discounts = 1.0 / np.log2(positions + 1)
    ideal_dcg = np.cumsum(discounts)
    cum_hits = np.cumsum(hits, axis=1)
    precision_at_rank = cum_hits / positions

    if item_popularity is not None:
        pop = np.asarray(item_popularity, dtype=np.float64)
        safe_top = np.where((top >= 0) & (top < len(pop)), top, 0)
        top_pop = np.where(top >= 0, pop[safe_top], np.nan)
        denom = max(int(n_popularity_users), 1)
        top_novelty = np.where(top >= 0, -np.log2(np.maximum(top_pop, 1.0) / denom), np.nan)

    results: Dict[str, float] = {}
    for k in ks:
        h = hits[:, :k]
        n_hits = h.sum(axis=1)
        ideal_k = np.minimum(n_relevant, k).astype(np.int64)

        dcg = (h * discounts[:k]).sum(axis=1)
        first = np.where(n_hits > 0, h.argmax(axis=1) + 1, 0)
        rr = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
        ap = (precision_at_rank[:, :k] * h).sum(axis=1) / ideal_k

        results[f"ndcg@{k}"] = _user_mean(dcg / ideal_dcg[ideal_k - 1], users)
        results[f"hit_rate@{k}"] = _user_mean((n_hits > 0).astype(np.float64), users)
        results[f"precision@{k}"] = _user_mean(n_hits / k, users)
        results[f"recall@{k}"] = _user_mean(n_hits / n_relevant, users)
        results[f"mrr@{k}"] = _user_mean(rr, users)
        results[f"map@{k}"] = _user_mean(ap, users)

        top_k = top[:, :k]
        recommended = top_k[top_k >= 0]
        _, exposure = np.unique(recommended, return_counts=True)
        catalog = max(int(n_catalog_items or 0), len(exposure), 1)
        results[f"coverage@{k}"] = float(len(exposure) / catalog)
        results[f"gini@{k}"] = _gini(np.concatenate([exposure, np.zeros(catalog - len(exposure))]))

        if item_popularity is not None:
            results[f"avg_popularity@{k}"] = _user_mean(np.nan_to_num(np.nanmean(top_pop[:, :k], axis=1)), users)
            results[f"novelty@{k}"] = _user_mean(np.nan_to_num(np.nanmean(top_novelty[:, :k], axis=1)), users)

    return results


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


@dataclass
class RankingResult:
    """Output of :func:`evaluate_rankings`.

    :ivar metrics: The model's metrics.
    :ivar baseline_metrics: Most-popular baseline on the same queries (empty without history).
    :ivar protocol: How the evaluation was run (persisted next to the metrics).
    :ivar queries: The queries that were evaluated (after dropping empty ones).
    :ivar top_items: Model top-``max(ks)`` item indices per evaluated query.
    :ivar top_scores: Scores of ``top_items``.
    :ivar n_candidates: Candidates ranked per evaluated query (after removing seen items).
    """

    metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    protocol: Dict[str, Any]
    queries: RankingQueries
    top_items: List[np.ndarray]
    top_scores: List[np.ndarray]
    n_candidates: np.ndarray


def evaluate_rankings(
    queries: RankingQueries,
    scores: Sequence[Any],
    candidates: Sequence[np.ndarray],
    ks: Iterable[int] = DEFAULT_KS,
    history: Optional[InteractionHistory] = None,
    n_catalog_items: Optional[int] = None,
    candidate_source: str = "full_catalog",
    seed: int = 0,
) -> RankingResult:
    """Rank every query's candidates and compute model + baseline metrics.

    :param queries: Queries, aligned with ``scores`` / ``candidates``.
    :param scores: Model score per candidate, one 1-D array (or tensor) per query.
    :param candidates: Candidate item indices per query.
    :param ks: Cutoffs.
    :param history: Training interactions. Enables seen-item exclusion,
        the popularity metrics and the most-popular baseline.
    :param n_catalog_items: Catalog size (for coverage / gini).
    :param candidate_source: Label recorded in the protocol (``full_catalog``,
        ``available_items`` or the candidate provider's name).
    :param seed: Seed of the random tie-break.
    """
    if not (len(queries) == len(scores) == len(candidates)):
        raise ValueError("queries, scores and candidates must be aligned")
    ks = sorted(set(int(k) for k in ks))
    max_k = ks[-1]
    rng = np.random.default_rng(seed)
    baseline_rng = np.random.default_rng(seed + 1)

    seen = history.seen if history is not None else {}
    popularity = history.popularity if history is not None else None

    keep: List[int] = []
    relevant: List[np.ndarray] = []
    top_items: List[np.ndarray] = []
    top_scores: List[np.ndarray] = []
    baseline_items: List[np.ndarray] = []
    n_candidates: List[int] = []
    n_relevant_total = 0
    n_relevant_seen = 0

    for q in range(len(queries)):
        cand = np.asarray(candidates[q], dtype=np.int64)
        score = scores[q]
        score = score.detach().cpu().numpy() if hasattr(score, "detach") else np.asarray(score)
        user_seen = seen.get(int(queries.users[q]))
        rel = queries.relevant[q]
        n_relevant_total += len(rel)
        if user_seen is not None and len(user_seen):
            not_seen = ~np.isin(cand, user_seen)
            cand, score = cand[not_seen], score[not_seen]
            rel_kept = rel[~np.isin(rel, user_seen)]
            n_relevant_seen += len(rel) - len(rel_kept)
            rel = rel_kept
        if len(rel) == 0:
            continue

        items, item_scores = rank_candidates(score, cand, max_k, rng)
        keep.append(q)
        relevant.append(rel)
        top_items.append(items)
        top_scores.append(item_scores)
        n_candidates.append(len(cand))
        if popularity is not None:
            pop_scores = np.where(cand < len(popularity), popularity[np.minimum(cand, len(popularity) - 1)], 0.0)
            baseline_items.append(rank_candidates(pop_scores, cand, max_k, baseline_rng)[0])

    evaluated = queries.subset(np.asarray(keep, dtype=np.int64))
    metric_kwargs = dict(
        ks=ks,
        users=evaluated.users,
        n_catalog_items=n_catalog_items,
        item_popularity=popularity,
        n_popularity_users=history.n_users if history is not None else 0,
    )
    metrics = compute_ranking_metrics(top_items, relevant, **metric_kwargs)
    baseline = compute_ranking_metrics(baseline_items, relevant, **metric_kwargs) if popularity is not None else {}

    n_cand = np.asarray(n_candidates, dtype=np.int64)
    protocol = {
        "candidates": candidate_source,
        "exclude_seen": history is not None,
        "aggregation": "mean over queries per user, then over users",
        "tie_break": f"random (seed={seed})",
        "ks": ks,
        "n_test_rows": int(queries.n_rows),
        "n_positive_rows": int(queries.n_positive_rows),
        "n_queries": int(len(evaluated)),
        "n_queries_dropped": int(len(queries) - len(evaluated)),
        "n_users": int(len(np.unique(evaluated.users))),
        "n_relevant": int(n_relevant_total - n_relevant_seen),
        "n_relevant_seen_dropped": int(n_relevant_seen),
        "avg_candidates": float(n_cand.mean()) if len(n_cand) else 0.0,
        "n_catalog_items": int(n_catalog_items) if n_catalog_items else None,
    }
    return RankingResult(
        metrics=metrics,
        baseline_metrics=baseline,
        protocol=protocol,
        queries=evaluated,
        top_items=top_items,
        top_scores=top_scores,
        n_candidates=n_cand,
    )
