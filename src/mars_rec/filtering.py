"""Candidate eligibility / pre-filtering layer for inference and eval.

A ``CandidateProvider`` decides, given a row of inference/eval input and the
full catalog of indexed items, **which subset of items is eligible for
scoring**. It generalizes the legacy ``ProjectConfig.available_items_column_name``
mechanism (which is a static per-row column populated offline) into a
plug-in providers usable in three places:

  - training-time eval     (``_BaseModelTraining._get_candidate_items``)
  - CE inference           (``inference.predict_top_k`` — masks logits)
  - BCE inference          (``inference.score_candidates_bce`` — restricts catalog)

The same provider in all three paths guarantees the offline ranking metric
mirrors what will be submitted.

Default behaviour: when no provider is configured, every consumer keeps its
legacy code path, so adding this module is a strictly opt-in change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


__all__ = [
    "CandidateProvider",
    "CandidateFilter",
    "FullCatalogProvider",
    "AvailableItemsColumnProvider",
    "CompositeProvider",
]


class CandidateProvider(ABC):
    """Decides which catalog items are eligible for scoring on a given row.

    Implementations should be cheap; ``get_candidates`` is called once per
    row at scoring time. Heavy precomputation belongs in ``__init__`` (e.g.
    aligning lookups to ``full_catalog`` ordering).
    """

    @abstractmethod
    def get_candidates(
        self,
        row: Dict[str, Any],
        full_catalog: np.ndarray,
    ) -> np.ndarray:
        """Return a 1-D ``int64`` array of eligible item indices.

        Args:
            row: per-observation context dict (raw or framework-indexed).
            full_catalog: int64 indices for every catalog item, sorted, with
                no PAD/OOV slots.

        Returns:
            Subset of ``full_catalog`` (same dtype). May be the input array
            itself when nothing is filtered out.
        """

    def name(self) -> str:
        """Human-readable identifier used in logs and diagnostics (defaults to the class name)."""
        return type(self).__name__


class CandidateFilter(ABC):
    """Boolean-mask producer aligned to ``full_catalog`` indexing.

    Used inside ``CompositeProvider`` to chain multiple eligibility rules.
    Each filter sees the full catalog (not the post-mask survivors) so the
    produced mask is always positionally aligned with the original array;
    this lets the composer combine masks via bitwise AND without re-indexing.
    """

    @abstractmethod
    def mask(
        self,
        row: Dict[str, Any],
        full_catalog: np.ndarray,
    ) -> np.ndarray:
        """Return a boolean mask of length ``len(full_catalog)``."""

    def score(
        self,
        row: Dict[str, Any],
        full_catalog: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return per-catalog float scores, or ``None`` if mask-only.

        Used by ``CompositeProvider.rank`` to ranking candidates without a
        downstream model. Filters that already compute scores internally
        (cosine, BM25) should override this; pure-boolean filters
        (year_window, goal_conditioned, …) leave it as ``None``.
        """
        return None

    def name(self) -> str:
        """Human-readable identifier used in logs and diagnostics (defaults to the class name)."""
        return type(self).__name__


# ---------------------------------------------------------------------------
# Built-in providers
# ---------------------------------------------------------------------------


class FullCatalogProvider(CandidateProvider):
    """No-op provider: every catalog item is eligible.

    Used as the default fallback for ``CompositeProvider`` when filters
    over-prune below ``min_pool``.
    """

    def get_candidates(self, row, full_catalog):
        """Return ``full_catalog`` unchanged."""
        return full_catalog

    def rank(
        self,
        row: Dict[str, Any],
        full_catalog: np.ndarray,
        top_k: int,
        fusion: str = "rrf",
        popularity_fallback: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Top-K by popularity (or catalog order when no popularity given)."""
        if top_k <= 0 or len(full_catalog) == 0:
            return np.asarray([], dtype=np.int64)
        if popularity_fallback:
            scores = np.asarray(
                [float(popularity_fallback.get(int(i), 0.0)) for i in full_catalog],
                dtype=np.float64,
            )
            k = min(int(top_k), len(full_catalog))
            top = np.argpartition(-scores, k - 1)[:k]
            order = top[np.argsort(-scores[top], kind="stable")]
            return np.asarray(full_catalog[order], dtype=np.int64)
        return np.asarray(full_catalog[: int(top_k)], dtype=np.int64)


class AvailableItemsColumnProvider(CandidateProvider):
    """Reads eligible idx from a row column.

    Backward-compatible with the legacy ``available_items_column_name``
    behaviour used by training-time eval today: a per-row list of integer
    item indices (already in framework-indexed space).

    If the column is missing or empty the full catalog is returned, so this
    provider is safe to wire in even when not every row carries the column.
    """

    def __init__(self, column_name: str):
        self.column_name = column_name

    def get_candidates(self, row, full_catalog):
        """Return the indices stored in ``row[column_name]``, or ``full_catalog`` when missing/empty."""
        items = row.get(self.column_name)
        if items is None:
            return full_catalog
        # Tolerate scalar / numpy / list — coerce to int64 array.
        arr = np.asarray(list(items), dtype=np.int64)
        if arr.size == 0:
            return full_catalog
        return arr


class CompositeProvider(CandidateProvider):
    """Combine a chain of ``CandidateFilter``s into a candidate pool.

    Two composition modes, applied in order:

    - ``filters`` (intersection): each filter's mask is AND-ed together.
      Conservative — every filter must agree.
    - ``union_filters`` (additive): each mask is OR-ed *into* the result.
      Permissive — a candidate that any union filter likes is included
      even if some intersection filter rejects it. Useful for "always
      include items from the user's last category" style boosts.

    Final mask = (AND over filters) ∪ (OR over union_filters). When
    ``filters`` is empty the AND part defaults to all-True (i.e. starts
    from the full catalog). When ``union_filters`` is empty the OR part is
    skipped (behavior identical to the legacy AND-only chain).

    If the final pool is smaller than ``min_pool``, the ``fallback`` provider
    is used instead — guards against over-pruning that would crater recall.
    """

    def __init__(
        self,
        filters: Sequence[CandidateFilter],
        min_pool: int = 0,
        fallback: Optional[CandidateProvider] = None,
        union_filters: Optional[Sequence[CandidateFilter]] = None,
    ):
        self.filters = list(filters)
        self.union_filters = list(union_filters or [])
        self.min_pool = int(min_pool)
        self.fallback = fallback or FullCatalogProvider()

    def get_candidates(self, row, full_catalog):
        if not self.filters and not self.union_filters:
            return full_catalog
        n = len(full_catalog)
        if self.filters:
            mask = np.ones(n, dtype=bool)
            for f in self.filters:
                mask &= f.mask(row, full_catalog)
                if not mask.any() and not self.union_filters:
                    break
        else:
            mask = np.zeros(n, dtype=bool) if self.union_filters else np.ones(n, dtype=bool)
        for f in self.union_filters:
            mask |= f.mask(row, full_catalog)
        if int(mask.sum()) < self.min_pool:
            return self.fallback.get_candidates(row, full_catalog)
        return full_catalog[mask]

    def rank(
        self,
        row: Dict[str, Any],
        full_catalog: np.ndarray,
        top_k: int,
        fusion: str = "rrf",
        popularity_fallback: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Return top-K framework indices sorted desc by fused filter scores.

        Algorithm:
            1. Apply intersect/union masks as in ``get_candidates``, including
               the ``min_pool`` fallback (full catalog when pool too small).
            2. Collect ``score()`` outputs from all filters that return one.
            3. Fuse via ``fusion`` strategy (``rrf`` / ``sum`` / ``max``).
            4. If no scoring filter contributed, fall back to
               ``popularity_fallback`` (idx → popularity dict); finally to
               catalog order.
        """
        eligible = self.get_candidates(row, full_catalog)
        if eligible is None or len(eligible) == 0:
            return np.asarray([], dtype=np.int64)
        if top_k <= 0:
            return np.asarray([], dtype=np.int64)

        # Collect per-filter scores aligned to ``eligible``.
        # Each filter.score(row, full_catalog) is per-full-catalog; we index it
        # down to the eligible subset positions so fusion works on a common
        # shape ``[len(eligible)]``.
        all_filters: List[CandidateFilter] = list(self.filters) + list(self.union_filters)
        # Build a positional mapping eligible→indices into full_catalog.
        # Since eligible == full_catalog[mask], the relative order is preserved
        # and we just need to know each eligible item's position in full_catalog
        # to index full-catalog-shaped score arrays.
        # A boolean mask over full_catalog handles both the masked and the
        # min_pool-fallback (full catalog) cases (np.in1d removed in numpy 2.x).
        n_full = len(full_catalog)
        keep = np.isin(full_catalog, eligible)

        scored: List[np.ndarray] = []
        for f in all_filters:
            s = f.score(row, full_catalog)
            if s is None:
                continue
            s = np.asarray(s, dtype=np.float64)
            if s.shape[0] != n_full:
                continue
            scored.append(s[keep])

        n_eligible = int(keep.sum())
        if n_eligible == 0:
            return np.asarray([], dtype=np.int64)

        if scored:
            fused = _fuse_scores(scored, fusion=fusion)
        elif popularity_fallback:
            # Tiebreaker by popularity. Missing entries get 0.
            elig_idx = full_catalog[keep]
            fused = np.asarray(
                [float(popularity_fallback.get(int(i), 0.0)) for i in elig_idx],
                dtype=np.float64,
            )
        else:
            # No score signal and no popularity → preserve catalog order.
            fused = -np.arange(n_eligible, dtype=np.float64)

        k = min(int(top_k), n_eligible)  # >= 1: both checked above
        # argpartition for speed; then sort the top-K slice desc.
        if k >= n_eligible:
            order = np.argsort(-fused, kind="stable")
        else:
            top_unsorted = np.argpartition(-fused, k - 1)[:k]
            order = top_unsorted[np.argsort(-fused[top_unsorted], kind="stable")]
        elig_idx = full_catalog[keep]
        return np.asarray(elig_idx[order[:k]], dtype=np.int64)

    def name(self) -> str:
        parts = []
        if self.filters:
            parts.append(" & ".join(f.name() for f in self.filters))
        if self.union_filters:
            parts.append("+(" + " | ".join(f.name() for f in self.union_filters) + ")")
        body = " ".join(parts) if parts else "<empty>"
        return f"Composite({body}, min_pool={self.min_pool})"


def _fuse_scores(scores_list: List[np.ndarray], fusion: str) -> np.ndarray:
    """Combine multiple per-eligible score arrays into one.

    All arrays are shape ``[n_eligible]``. Supported fusions:
      - ``rrf``: reciprocal-rank fusion (k=60). Robust to score-scale
        differences; identical to the single source when only one filter
        scores.
      - ``sum``: min-max normalize each source to [0,1], then sum.
      - ``max``: per-item max over sources (after min-max normalization).
    """
    if not scores_list:
        return np.zeros(0, dtype=np.float64)
    if fusion == "rrf":
        K_RRF = 60.0
        fused = np.zeros_like(scores_list[0], dtype=np.float64)
        for s in scores_list:
            # rank by descending score; rank 0 = best
            order = np.argsort(-s, kind="stable")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(s))
            fused += 1.0 / (K_RRF + ranks.astype(np.float64))
        return fused
    if fusion in ("sum", "max"):
        norms = []
        for s in scores_list:
            lo, hi = float(np.min(s)), float(np.max(s))
            if hi - lo < 1e-12:
                norms.append(np.zeros_like(s, dtype=np.float64))
            else:
                norms.append(((s - lo) / (hi - lo)).astype(np.float64))
        if fusion == "sum":
            return np.sum(np.stack(norms, axis=0), axis=0)
        return np.max(np.stack(norms, axis=0), axis=0)
    raise ValueError(f"Unknown fusion strategy: {fusion!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_candidate_provider(
    project_config,
    index_mapping: Dict[str, Dict[Any, int]],
    dataset_extras: Optional[Dict[str, Any]] = None,
) -> Optional[CandidateProvider]:
    """Instantiate the project-configured provider, if any.

    Returns ``None`` when neither a ``candidate_provider_factory`` nor a
    legacy ``available_items_column_name`` is configured, so callers can
    keep their current code paths via a single ``if provider is None`` guard.
    """
    factory = getattr(project_config, "candidate_provider_factory", None)
    if factory is not None:
        return factory(project_config, index_mapping, dataset_extras or {})
    col = getattr(project_config, "available_items_column_name", None)
    if col:
        return AvailableItemsColumnProvider(col)
    return None
