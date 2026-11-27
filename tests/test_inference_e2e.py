"""End-to-end inference tests for ``mars_rec.inference``.

Complements ``test_inference_contract.py`` (which tests each function in
isolation) by exercising the full ``run_inference`` orchestrator —
parquet read → preprocess → predict_top_k → reverse map → result.json
— plus the lower-level scoring helpers ``_collect_logits_and_targets``
and ``score_candidates_bce``.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from mars_rec.data.task import BasePrepareDataFrames
from mars_rec.inference import (
    _collect_logits_and_targets,
    run_inference,
    score_candidates_bce,
)
from mars_rec.meta_config import Column, IOType, ProjectConfig
from mars_rec.model.base_model import LogisticRegression


# ---------------------------------------------------------------------------
# Fixture project — must be importable via dotted path
# ---------------------------------------------------------------------------


class _E2EPrep(BasePrepareDataFrames):
    @property
    def dataset_dir(self):  # pragma: no cover
        return "/tmp"

    @property
    def read_data_frame_path(self):  # pragma: no cover
        return None


e2e_config = ProjectConfig(
    base_dir="/tmp/inf_e2e",
    prepare_data_frames_task=_E2EPrep,
    dataset_class=MagicMock(),
    user_column=Column("user_id", IOType.INDEXABLE),
    item_column=Column("item_id", IOType.INDEXABLE),
    other_input_columns=[],
    output_column=Column("label", IOType.NUMBER),
)


def _stage_task_dir(task_dir, n_factors: int = 4):
    """Write params.json + index_mapping.pkl + Lightning-style checkpoint."""
    idx = {
        "user_id": defaultdict(int, {"u1": 3, "u2": 4, "u3": 5}),
        "item_id": defaultdict(int, {"i1": 3, "i2": 4, "i3": 5, "i4": 6}),
    }
    model = LogisticRegression(e2e_config, idx, n_factors=n_factors)
    # Pin weights so reverse-mapping is deterministic across rows.
    with torch.no_grad():
        torch.manual_seed(0)
        for p in model.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.1)

    params = {
        "project": __name__ + ".e2e_config",
        "recommender_module_class": ("mars_rec.model.base_model.LogisticRegression"),
        "recommender_extra_params": {"n_factors": n_factors},
    }
    with open(task_dir / "params.json", "w") as f:
        json.dump(params, f)
    with open(task_dir / "index_mapping.pkl", "wb") as f:
        pickle.dump(idx, f)
    prefixed = {"module." + k: v for k, v in model.state_dict().items()}
    torch.save({"state_dict": prefixed}, task_dir / "weights_epoch=00.ckpt")
    return idx, model


# ---------------------------------------------------------------------------
# run_inference — full orchestrator
# ---------------------------------------------------------------------------


class TestRunInferenceOrchestrator:
    def test_writes_result_json_with_expected_schema(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        _stage_task_dir(task_dir)

        # Raw input: 3 rows, no label column (typical for blind test sets).
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "item_id": ["i1", "i2", "i3"],
            }
        )
        data_path = tmp_path / "inputs.parquet"
        df.to_parquet(data_path)

        output_path = tmp_path / "result.json"
        result = run_inference(
            task_dir=str(task_dir),
            data_path=str(data_path),
            output_path=str(output_path),
            batch_size=2,
            k=5,
            device="cpu",
        )

        # The function returns the dict it writes.
        assert set(result.keys()) == {"top10s", "user", "time"}
        assert result["user"] == ["u1", "u2", "u3"]
        assert len(result["top10s"]) == 3
        # File on disk matches.
        with open(output_path) as f:
            disk = json.load(f)
        assert disk["user"] == result["user"]

    def test_reads_csv_when_data_path_is_csv(self, tmp_path):
        """The orchestrator dispatches on file extension — .parquet vs other."""
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        _stage_task_dir(task_dir)

        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "item_id": ["i1", "i2"],
            }
        )
        data_path = tmp_path / "inputs.csv"
        df.to_csv(data_path, index=False)

        result = run_inference(
            task_dir=str(task_dir),
            data_path=str(data_path),
            output_path=str(tmp_path / "out.json"),
            batch_size=2,
            k=3,
            device="cpu",
        )
        assert len(result["user"]) == 2


# ---------------------------------------------------------------------------
# _collect_logits_and_targets
# ---------------------------------------------------------------------------


class TestCollectLogitsAndTargets:
    def test_shapes_match_dataset_size(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        idx, model = _stage_task_dir(task_dir)

        # Build a tiny dataset directly with the framework's class.
        from mars_rec.data.dataset import (
            InteractionsDataset,
            preprocess_interactions_data_frame,
        )
        from mars_rec.utils.index_mapping import transform_with_indexing

        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3", "u1"],
                "item_id": ["i1", "i2", "i3", "i4"],
                "label": [1.0, 0.0, 1.0, 0.0],
            }
        )
        df = preprocess_interactions_data_frame(df, e2e_config)
        transform_with_indexing(df, idx, e2e_config)
        ds = InteractionsDataset(
            data_frame=df,
            embeddings_for_metadata=None,
            project_config=e2e_config,
            index_mapping=idx,
        )

        logits, targets = _collect_logits_and_targets(
            model,
            ds,
            batch_size=2,
            device="cpu",
        )
        assert logits.shape[0] == 4
        assert targets.shape[0] == 4
        # LogisticRegression: [N, 1]; binary label: [N]
        assert logits.dim() == 2 and logits.size(1) == 1


# ---------------------------------------------------------------------------
# score_candidates_bce
# ---------------------------------------------------------------------------


class TestScoreCandidatesBCE:
    def test_returns_per_row_scores_for_full_catalog(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        idx, model = _stage_task_dir(task_dir)

        # Two input rows; full catalog = 4 items.
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "item_id": ["i1", "i2"],
                "label": [1.0, 0.0],
            }
        )
        from mars_rec.data.dataset import preprocess_interactions_data_frame
        from mars_rec.utils.index_mapping import transform_with_indexing

        df = preprocess_interactions_data_frame(df, e2e_config)
        transform_with_indexing(df, idx, e2e_config)
        full_catalog = np.array([3, 4, 5, 6], dtype=np.int64)

        turn_scores, eligible_per_row = score_candidates_bce(
            model,
            df,
            e2e_config,
            idx,
            full_catalog,
            batch_size=32,
            turn_batch_size=2,
            device="cpu",
            show_progress=False,
        )
        assert len(turn_scores) == 2
        assert len(eligible_per_row) == 2
        # No candidate_provider → eligible is the full catalog for every row.
        for elig in eligible_per_row:
            np.testing.assert_array_equal(elig, full_catalog)
        # Each score tensor matches the eligibility cardinality.
        for s, e in zip(turn_scores, eligible_per_row):
            assert s.shape[0] == len(e)
            # LogisticRegression has sigmoid, so scores ∈ [0, 1].
            assert ((s >= 0) & (s <= 1)).all()
