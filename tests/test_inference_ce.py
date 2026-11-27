"""Inference / evaluation on the next-item (CE) path — ``[batch, n_items]`` logits.

``test_inference_e2e.py`` drives the orchestrators with a point-wise model
(``[batch]`` scores). That never reaches the code that actually ranks the
catalog: top-k over logits, reverse mapping, ranking metrics in
``inference.run_evaluate``, and the ``score_candidates_ce`` / ``_emb``
helpers. These tests use the ``OracleCE`` fixture model, whose logit table
is pinned so each user's ground-truth item ranks first — every ranking
metric therefore has an exact expected value.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import torch

from mars_rec.filtering import CandidateProvider
from mars_rec.inference import (
    build_reverse_item_mapping,
    load_trained_model,
    predict_top_k,
    preprocess_raw_data,
    reserved_logit_indices,
    run_evaluate,
    run_inference,
    score_candidates_ce,
    score_candidates_emb,
)
from tests._fixtures.ce_project import OracleCE, ce_project_config, pin_oracle

# u1 -> i2, u2 -> i3, u3 -> i4 (raw ids); indices start at 3 (0/1/2 reserved).
USER_IDX = {"u1": 3, "u2": 4, "u3": 5}
ITEM_IDX = {"i1": 3, "i2": 4, "i3": 5, "i4": 6}
TRUTH = {"u1": "i2", "u2": "i3", "u3": "i4"}
FULL_CATALOG = np.array(sorted(ITEM_IDX.values()), dtype=np.int64)


def _index_mapping():
    from mars_rec.utils.index_mapping import create_index_mapping

    # Built by the real helper so the reserved slots (None/NaN→1, -1→2)
    # are exactly what training produces.
    return {
        "user_id": create_index_mapping(USER_IDX),
        "item_id": create_index_mapping(ITEM_IDX),
    }


@pytest.fixture
def ce_task_dir(tmp_path):
    """A task dir (params.json + index_mapping.pkl + checkpoint) for OracleCE."""
    idx = _index_mapping()
    assert dict((k, idx["item_id"][k]) for k in ITEM_IDX) == ITEM_IDX  # fixture sanity

    model = OracleCE(ce_project_config, idx)
    pin_oracle(model, {idx["user_id"][u]: idx["item_id"][i] for u, i in TRUTH.items()})

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    with open(task_dir / "params.json", "w") as f:
        json.dump(
            {
                "project": "tests._fixtures.ce_project.ce_project_config",
                "recommender_module_class": "tests._fixtures.ce_project.OracleCE",
                "recommender_extra_params": {},
                "loss_function": "ce",
            },
            f,
        )
    with open(task_dir / "index_mapping.pkl", "wb") as f:
        pickle.dump(idx, f)
    state = {"module." + k: v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state}, task_dir / "weights.ckpt")
    return task_dir


@pytest.fixture
def eval_csv(tmp_path):
    path = tmp_path / "eval.csv"
    pd.DataFrame({"user_id": list(TRUTH), "item_id": list(TRUTH.values())}).to_csv(path, index=False)
    return path


class _FixedProvider(CandidateProvider):
    """Same candidate set for every row."""

    def __init__(self, items):
        self.items = np.asarray(items, dtype=np.int64)

    def get_candidates(self, row, full_catalog):
        return self.items


# ---------------------------------------------------------------------------
# Reverse mapping / reserved slots
# ---------------------------------------------------------------------------


class TestReservedSlots:
    def test_reverse_mapping_contains_only_real_items(self):
        rev = build_reverse_item_mapping(_index_mapping(), "item_id")
        assert rev == {v: k for k, v in ITEM_IDX.items()}

    def test_reserved_logit_indices_cover_oov_nan_and_pad(self):
        rev = build_reverse_item_mapping(_index_mapping(), "item_id")
        assert reserved_logit_indices(rev, n_classes=7) == [0, 1, 2]


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------


class TestRunInferenceCE:
    def test_top1_is_ground_truth(self, ce_task_dir, eval_csv, tmp_path):
        result = run_inference(str(ce_task_dir), str(eval_csv), str(tmp_path / "r.json"), k=1)
        assert result["user"] == list(TRUTH)
        assert result["top10s"] == [[TRUTH[u]] for u in TRUTH]

    def test_returns_k_real_items_per_user(self, ce_task_dir, eval_csv, tmp_path):
        """Regression: reserved slots (OOV/NaN/PAD) used to win top-k
        positions and were then dropped by the reverse map, so users got
        fewer than ``k`` items — or the PAD token "-1" as an item. The
        oracle gives index 0 the second-highest logit on purpose."""
        result = run_inference(str(ce_task_dir), str(eval_csv), str(tmp_path / "r.json"), k=4)
        for user, items in zip(result["user"], result["top10s"]):
            assert len(items) == 4
            assert set(items) == set(ITEM_IDX)
            assert items[0] == TRUTH[user]

    def test_k_larger_than_catalog_returns_whole_catalog(self, ce_task_dir, eval_csv, tmp_path):
        result = run_inference(str(ce_task_dir), str(eval_csv), str(tmp_path / "r.json"), k=50)
        for items in result["top10s"]:
            assert sorted(items) == sorted(ITEM_IDX)


# ---------------------------------------------------------------------------
# inference.run_evaluate
# ---------------------------------------------------------------------------


class TestRunEvaluateCE:
    def test_oracle_gets_perfect_ranking_metrics(self, ce_task_dir, eval_csv, tmp_path):
        """Regression: ground truth was read from the *indexed* frame
        ("4") and compared with *raw* ids ("i2"), so every metric was 0.0
        no matter how good the model was."""
        out = tmp_path / "eval.json"
        result = run_evaluate(str(ce_task_dir), data_path=str(eval_csv), ranking_k=[1, 3], output_path=str(out))

        assert result["n_samples"] == len(TRUTH)
        rm = result["ranking_metrics"]
        for k in (1, 3):
            for m in ("ndcg", "hit_rate", "recall", "mrr", "map"):
                assert rm[f"{m}@{k}"] == 1.0, f"{m}@{k}"
            assert rm[f"precision@{k}"] == pytest.approx(1 / k)
        # Each user gets a different item on top: 3 of the 4 items recommended at k=1.
        assert rm["coverage@1"] == pytest.approx(3 / 4)
        # No training history in this standalone path → no popularity metrics.
        assert "novelty@1" not in rm
        assert result["ranking_protocol"]["n_queries"] == len(TRUTH)
        assert result["ranking_protocol"]["exclude_seen"] is False
        assert result["roc_auc_metrics"] == {"roc_auc_ovr": pytest.approx(1.0)}
        with open(out) as f:
            assert json.load(f)["ranking_metrics"] == result["ranking_metrics"]

    def test_wrong_ground_truth_scores_by_rank(self, ce_task_dir, tmp_path):
        """Every user's target is i1, which the oracle ranks 2nd for every
        user (right after their pinned item: i1 has the highest base logit
        among the real items)."""
        path = tmp_path / "eval.csv"
        pd.DataFrame({"user_id": list(TRUTH), "item_id": ["i1"] * 3}).to_csv(path, index=False)

        result = run_evaluate(str(ce_task_dir), data_path=str(path), ranking_k=[1, 2], output_path=str(tmp_path / "e"))
        rm = result["ranking_metrics"]
        assert rm["hit_rate@1"] == 0.0
        assert rm["hit_rate@2"] == 1.0
        assert rm["mrr@2"] == pytest.approx(0.5)
        assert rm["ndcg@2"] == pytest.approx(1 / np.log2(3))

    def test_unknown_target_counts_as_miss(self, ce_task_dir, tmp_path):
        path = tmp_path / "eval.csv"
        pd.DataFrame({"user_id": ["u1"], "item_id": ["never-seen"]}).to_csv(path, index=False)
        result = run_evaluate(str(ce_task_dir), data_path=str(path), ranking_k=[4], output_path=str(tmp_path / "e"))
        assert result["ranking_metrics"]["hit_rate@4"] == 0.0


# ---------------------------------------------------------------------------
# predict_top_k / score_candidates_* with a CandidateProvider
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded(ce_task_dir, eval_csv):
    model, config, idx, _ = load_trained_model(str(ce_task_dir))
    df, dataset = preprocess_raw_data(pd.read_csv(eval_csv), config, idx)
    return model, config, idx, df, dataset


class TestPredictTopKCE:
    def test_candidate_provider_masks_non_eligible(self, loaded):
        model, _, _, df, dataset = loaded
        # Only i1 and i3 are eligible → u2 (truth i3) keeps its item, others get i1/i3.
        provider = _FixedProvider([ITEM_IDX["i1"], ITEM_IDX["i3"]])
        indices, scores = predict_top_k(
            model, dataset, k=2, candidate_provider=provider, full_catalog=FULL_CATALOG, rows=df.to_dict("records")
        )
        assert indices.shape == (3, 2)
        for row in indices:
            assert set(row.tolist()) == {ITEM_IDX["i1"], ITEM_IDX["i3"]}
        assert indices[1, 0] == ITEM_IDX["i3"]
        assert np.isfinite(scores).all()

    def test_candidate_provider_requires_catalog_and_rows(self, loaded):
        model, _, _, _, dataset = loaded
        with pytest.raises(ValueError, match="full_catalog and rows"):
            predict_top_k(model, dataset, candidate_provider=_FixedProvider([3]))

    def test_exclude_indices_never_returned(self, loaded):
        model, _, _, _, dataset = loaded
        indices, _ = predict_top_k(model, dataset, k=3, exclude_indices=[0, 1, 2, ITEM_IDX["i1"]])
        assert not set(indices.ravel().tolist()) & {0, 1, 2, ITEM_IDX["i1"]}


class TestScoreCandidatesCE:
    def test_full_catalog_scores_rank_truth_first(self, loaded):
        model, config, idx, df, _ = loaded
        scores, eligible = score_candidates_ce(model, df, config, idx, FULL_CATALOG, show_progress=False)
        assert len(scores) == len(eligible) == len(TRUTH)
        for user, s, e in zip(TRUTH, scores, eligible):
            np.testing.assert_array_equal(e, FULL_CATALOG)
            assert e[int(torch.argmax(s))] == ITEM_IDX[TRUTH[user]]

    def test_provider_restricts_and_clips_to_vocab(self, loaded):
        model, config, idx, df, _ = loaded
        # 999 is outside the vocabulary and must be clipped away.
        provider = _FixedProvider([ITEM_IDX["i2"], ITEM_IDX["i4"], 999])
        scores, eligible = score_candidates_ce(
            model, df, config, idx, FULL_CATALOG, candidate_provider=provider, show_progress=False
        )
        for s, e in zip(scores, eligible):
            assert e.tolist() == [ITEM_IDX["i2"], ITEM_IDX["i4"]]
            assert s.shape == (2,)

    def test_rejects_pointwise_output(self, loaded):
        _, config, idx, df, _ = loaded

        class _Pointwise(torch.nn.Module):
            def forward(self, user_ids):
                return torch.zeros(user_ids.shape[0])

        with pytest.raises(ValueError, match=r"\[B, n_items\]"):
            score_candidates_ce(_Pointwise(), df, config, idx, FULL_CATALOG, show_progress=False)


class TestScoreCandidatesEmb:
    def test_dot_product_matches_logits(self, loaded):
        """OracleCE's target embedding is one-hot, so ``pred @ item.T`` is
        exactly the logit of each candidate."""
        model, config, idx, df, _ = loaded
        scores, eligible = score_candidates_emb(model, df, config, idx, FULL_CATALOG, show_progress=False)
        with torch.no_grad():
            logits = model(torch.as_tensor(df["user_id"].to_numpy(), dtype=torch.long))
        for i, (s, e) in enumerate(zip(scores, eligible)):
            torch.testing.assert_close(s, logits[i, torch.as_tensor(e)])

    def test_with_provider(self, loaded):
        model, config, idx, df, _ = loaded
        provider = _FixedProvider([ITEM_IDX["i3"]])
        scores, eligible = score_candidates_emb(
            model, df, config, idx, FULL_CATALOG, candidate_provider=provider, show_progress=False
        )
        assert all(e.tolist() == [ITEM_IDX["i3"]] for e in eligible)
        assert all(s.shape == (1,) for s in scores)

    def test_requires_get_target_emb(self, loaded):
        _, config, idx, df, _ = loaded
        with pytest.raises(AttributeError, match="get_target_emb"):
            score_candidates_emb(torch.nn.Linear(1, 1), df, config, idx, FULL_CATALOG, show_progress=False)
