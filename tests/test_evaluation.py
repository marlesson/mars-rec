"""Unit tests for ``mars_rec.evaluation`` — every value is worked out by hand."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mars_rec.evaluation import (
    InteractionHistory,
    build_ranking_queries,
    compute_ranking_metrics,
    evaluate_rankings,
    positive_mask,
    rank_candidates,
    ranking_metric_names,
)


def _arr(*values):
    return np.asarray(values, dtype=np.int64)


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------


class TestAccuracyMetrics:
    def test_single_relevant_at_rank_two(self):
        m = compute_ranking_metrics([_arr(10, 20, 30)], [_arr(20)], ks=[1, 3])
        assert m["hit_rate@1"] == 0.0
        assert m["hit_rate@3"] == 1.0
        assert m["precision@3"] == pytest.approx(1 / 3)
        assert m["recall@3"] == 1.0
        assert m["mrr@3"] == pytest.approx(0.5)
        assert m["ndcg@3"] == pytest.approx(1 / math.log2(3))
        assert m["map@3"] == pytest.approx(0.5)

    def test_multiple_relevant(self):
        # Relevant {1, 3, 9}; ranked 1 . 3 . . → hits at ranks 1 and 3.
        m = compute_ranking_metrics([_arr(1, 2, 3, 4, 5)], [_arr(1, 3, 9)], ks=[5])
        assert m["hit_rate@5"] == 1.0
        assert m["precision@5"] == pytest.approx(2 / 5)
        assert m["recall@5"] == pytest.approx(2 / 3)
        assert m["mrr@5"] == 1.0
        dcg = 1 + 1 / math.log2(4)
        idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        assert m["ndcg@5"] == pytest.approx(dcg / idcg)
        # AP = (P@1 + P@3) / min(|R|, k) = (1 + 2/3) / 3
        assert m["map@5"] == pytest.approx((1 + 2 / 3) / 3)

    def test_map_differs_from_mrr_with_several_relevant(self):
        m = compute_ranking_metrics([_arr(1, 2, 3)], [_arr(2, 3)], ks=[3])
        assert m["mrr@3"] == pytest.approx(0.5)
        assert m["map@3"] == pytest.approx((1 / 2 + 2 / 3) / 2)

    def test_ideal_dcg_is_truncated_at_k(self):
        # More relevant items than k: a perfect top-2 still gets NDCG 1 and MAP 1.
        m = compute_ranking_metrics([_arr(1, 2)], [_arr(1, 2, 3, 4)], ks=[2])
        assert m["ndcg@2"] == pytest.approx(1.0)
        assert m["map@2"] == pytest.approx(1.0)
        assert m["recall@2"] == pytest.approx(0.5)

    def test_miss_is_zero(self):
        m = compute_ranking_metrics([_arr(1, 2)], [_arr(7)], ks=[2])
        assert all(m[f"{name}@2"] == 0.0 for name in ("ndcg", "hit_rate", "precision", "recall", "mrr", "map"))

    def test_short_list_counts_precision_over_k(self):
        m = compute_ranking_metrics([_arr(5)], [_arr(5)], ks=[4])
        assert m["precision@4"] == pytest.approx(0.25)
        assert m["hit_rate@4"] == 1.0

    def test_user_macro_average(self):
        # User 1 has two queries (hit, miss) → 0.5; user 2 one hit → 1.0. Mean = 0.75,
        # while the plain mean over queries would be 2/3.
        ranked = [_arr(1), _arr(2), _arr(3)]
        relevant = [_arr(1), _arr(9), _arr(3)]
        per_user = compute_ranking_metrics(ranked, relevant, ks=[1], users=np.array([1, 1, 2]))
        per_query = compute_ranking_metrics(ranked, relevant, ks=[1])
        assert per_user["hit_rate@1"] == pytest.approx(0.75)
        assert per_query["hit_rate@1"] == pytest.approx(2 / 3)

    def test_metrics_monotonic_in_k(self):
        rng = np.random.default_rng(0)
        ranked = [rng.permutation(50)[:20] for _ in range(30)]
        relevant = [rng.choice(50, size=3, replace=False) for _ in range(30)]
        m = compute_ranking_metrics(ranked, relevant, ks=[5, 10, 20])
        for name in ("hit_rate", "recall", "ndcg", "mrr", "coverage"):
            assert m[f"{name}@5"] <= m[f"{name}@10"] <= m[f"{name}@20"], name

    def test_empty_and_misaligned(self):
        assert compute_ranking_metrics([], [], ks=[5]) == {}
        with pytest.raises(ValueError):
            compute_ranking_metrics([_arr(1)], [], ks=[5])

    def test_metric_names_match_output(self):
        m = compute_ranking_metrics(
            [_arr(1, 2)], [_arr(1)], ks=[5, 10], item_popularity=np.ones(3), n_popularity_users=1
        )
        assert set(m) == set(ranking_metric_names([5, 10]))
        m = compute_ranking_metrics([_arr(1, 2)], [_arr(1)], ks=[5])
        assert set(m) == set(ranking_metric_names([5], with_popularity=False))


# ---------------------------------------------------------------------------
# Beyond-accuracy metrics
# ---------------------------------------------------------------------------


class TestBeyondAccuracy:
    def test_coverage(self):
        m = compute_ranking_metrics([_arr(1, 2), _arr(2, 3)], [_arr(1), _arr(1)], ks=[2], n_catalog_items=10)
        assert m["coverage@2"] == pytest.approx(3 / 10)

    def test_gini_even_vs_concentrated(self):
        even = compute_ranking_metrics([_arr(0), _arr(1)], [_arr(0), _arr(1)], ks=[1], n_catalog_items=2)
        concentrated = compute_ranking_metrics([_arr(0), _arr(0)], [_arr(0), _arr(0)], ks=[1], n_catalog_items=2)
        assert even["gini@1"] == pytest.approx(0.0)
        # Exposure [0, 2] over 2 items: G = (−1·0 + 1·2) / (2·2) = 0.5.
        assert concentrated["gini@1"] == pytest.approx(0.5)

    def test_popularity_and_novelty(self):
        popularity = np.array([0.0, 8.0, 2.0])
        m = compute_ranking_metrics([_arr(1, 2)], [_arr(1)], ks=[2], item_popularity=popularity, n_popularity_users=8)
        assert m["avg_popularity@2"] == pytest.approx(5.0)
        # -log2(8/8) = 0, -log2(2/8) = 2 → mean 1.
        assert m["novelty@2"] == pytest.approx(1.0)

    def test_novelty_of_unseen_item_is_bounded(self):
        m = compute_ranking_metrics([_arr(0)], [_arr(0)], ks=[1], item_popularity=np.zeros(1), n_popularity_users=4)
        assert m["novelty@1"] == pytest.approx(2.0)  # popularity floored at 1 → -log2(1/4)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestRankCandidates:
    def test_descending_order(self):
        items, scores = rank_candidates(np.array([0.1, 0.9, 0.5]), _arr(7, 8, 9), k=2, rng=np.random.default_rng(0))
        assert items.tolist() == [8, 9]
        assert scores.tolist() == [0.9, 0.5]

    def test_ties_are_not_broken_by_candidate_order(self):
        """A constant model must not get hits from where the item sits in the list."""
        firsts = {
            int(rank_candidates(np.zeros(4), _arr(1, 2, 3, 4), k=1, rng=np.random.default_rng(seed))[0][0])
            for seed in range(40)
        }
        assert firsts == {1, 2, 3, 4}

    def test_ties_are_deterministic_for_a_seed(self):
        a = rank_candidates(np.zeros(5), _arr(1, 2, 3, 4, 5), k=5, rng=np.random.default_rng(3))[0]
        b = rank_candidates(np.zeros(5), _arr(1, 2, 3, 4, 5), k=5, rng=np.random.default_rng(3))[0]
        assert a.tolist() == b.tolist()

    def test_nan_ranks_last(self):
        items, _ = rank_candidates(np.array([np.nan, 0.1]), _arr(1, 2), k=2, rng=np.random.default_rng(0))
        assert items.tolist() == [2, 1]

    def test_empty(self):
        items, scores = rank_candidates(np.array([]), _arr(), k=3, rng=np.random.default_rng(0))
        assert len(items) == len(scores) == 0


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestBuildQueries:
    def _df(self):
        return pd.DataFrame(
            {
                "user": [3, 3, 3, 4, 4],
                "item": [10, 11, 12, 10, 13],
                "label": [1.0, 0.0, 1.0, 1.0, 1.0],
                "hist": [[1], [1], [1], [5], [6]],
            }
        )

    def test_positive_mask(self):
        df = self._df()
        assert positive_mask(df, "item", "label").tolist() == [True, False, True, True, True]
        # Item-valued target: every row is ground truth.
        assert positive_mask(df, "item", "item").all()

    def test_groups_rows_with_the_same_context(self):
        q = build_ranking_queries(self._df(), "user", "item", "label")
        assert len(q) == 2
        assert q.users.tolist() == [3, 4]
        assert [r.tolist() for r in q.relevant] == [[10, 12], [10, 13]]
        assert q.n_rows == 5 and q.n_positive_rows == 4

    def test_context_columns_split_queries(self):
        """A per-row history (sequential models) makes every positive row its own query."""
        q = build_ranking_queries(self._df(), "user", "item", "label", context_columns=["hist"])
        assert [r.tolist() for r in q.relevant] == [[10, 12], [10], [13]]
        assert q.positions.tolist() == [0, 3, 4]

    def test_negative_rows_are_not_ground_truth(self):
        q = build_ranking_queries(self._df(), "user", "item", "label")
        assert 11 not in np.concatenate(q.relevant)

    def test_no_positive_rows(self):
        df = self._df().assign(label=0.0)
        assert len(build_ranking_queries(df, "user", "item", "label")) == 0

    def test_sample(self):
        q = build_ranking_queries(self._df(), "user", "item", "label", context_columns=["hist"])
        assert len(q.sample(None)) == 3
        assert len(q.sample(2, seed=1)) == 2
        assert q.sample(2, seed=1).positions.tolist() == q.sample(2, seed=1).positions.tolist()


# ---------------------------------------------------------------------------
# History + end-to-end
# ---------------------------------------------------------------------------


class TestInteractionHistory:
    def test_seen_and_popularity(self):
        h = InteractionHistory.from_interactions(
            users=_arr(3, 3, 4, 4, 4),
            items=_arr(10, 11, 10, 10, 12),
            positive=np.array([True, False, True, True, False]),
            n_items=13,
        )
        assert h.seen[3].tolist() == [10, 11]
        assert h.seen[4].tolist() == [10, 12]
        # Item 10: two distinct users with a positive; the duplicate (4, 10) counts once.
        assert h.popularity[10] == 2 and h.popularity[11] == 0 and h.popularity[12] == 0
        assert h.n_users == 2


class TestEvaluateRankings:
    def _queries(self):
        df = pd.DataFrame({"user": [3, 4], "item": [12, 11], "label": [1.0, 1.0]})
        return build_ranking_queries(df, "user", "item", "label")

    def test_seen_items_are_excluded_from_candidates(self):
        catalog = _arr(10, 11, 12)
        # The model loves item 10, which user 3 already saw: it must not take the top slot.
        scores = [np.array([0.9, 0.1, 0.5]), np.array([0.9, 0.5, 0.1])]
        history = InteractionHistory.from_interactions(_arr(3), _arr(10), np.array([True]), n_items=13)
        result = evaluate_rankings(
            self._queries(), scores, [catalog, catalog], ks=[1], history=history, n_catalog_items=3
        )
        assert result.top_items[0].tolist() == [12]
        assert result.top_items[1].tolist() == [10]
        assert result.metrics["hit_rate@1"] == pytest.approx(0.5)
        assert result.n_candidates.tolist() == [2, 3]

    def test_relevant_item_seen_in_training_is_dropped(self):
        catalog = _arr(10, 11, 12)
        history = InteractionHistory.from_interactions(_arr(3), _arr(12), np.array([True]), n_items=13)
        scores = [np.zeros(3), np.array([0.0, 1.0, 0.0])]
        result = evaluate_rankings(self._queries(), scores, [catalog, catalog], ks=[1], history=history)
        assert result.protocol["n_relevant_seen_dropped"] == 1
        assert result.protocol["n_queries"] == 1 and result.protocol["n_queries_dropped"] == 1
        assert result.metrics["hit_rate@1"] == 1.0

    def test_popularity_baseline(self):
        catalog = _arr(10, 11, 12)
        # Item 12 is the most popular; user 3's relevant item is 12.
        history = InteractionHistory.from_interactions(
            _arr(7, 8, 9), _arr(12, 12, 11), np.array([True, True, True]), n_items=13
        )
        scores = [np.zeros(3), np.zeros(3)]
        result = evaluate_rankings(self._queries(), scores, [catalog, catalog], ks=[1], history=history)
        assert result.baseline_metrics["hit_rate@1"] == pytest.approx(0.5)
        assert "avg_popularity@1" in result.metrics

    def test_without_history(self):
        catalog = _arr(10, 11, 12)
        scores = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])]
        result = evaluate_rankings(self._queries(), scores, [catalog, catalog], ks=[1, 2], n_catalog_items=3)
        assert result.metrics["ndcg@1"] == 1.0
        assert result.baseline_metrics == {}
        assert "novelty@1" not in result.metrics
        assert result.protocol["exclude_seen"] is False
        assert result.protocol["candidates"] == "full_catalog"

    def test_accepts_torch_scores(self):
        torch = pytest.importorskip("torch")
        catalog = _arr(10, 11, 12)
        scores = [torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 1.0, 0.0])]
        result = evaluate_rankings(self._queries(), scores, [catalog, catalog], ks=[1])
        assert result.metrics["hit_rate@1"] == 1.0

    def test_misaligned_inputs_raise(self):
        with pytest.raises(ValueError):
            evaluate_rankings(self._queries(), [np.zeros(3)], [_arr(1, 2, 3)], ks=[1])
