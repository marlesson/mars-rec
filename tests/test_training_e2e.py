"""End-to-end Luigi lifecycle for ``SupervisedModelTraining``.

This is the heavy-weight counterpart to ``test_training_lifecycle.py``:
it actually dispatches the full Luigi task — prep → indexing → training →
evaluation → ranking metrics → metadata — against a fixture project under
``tests/_fixtures/dummy_project.py``, and then drives every consumer of the
resulting task directory from it:

- ``simulation.training.run_evaluate`` (original splits + external CSV);
- ``inference.run_evaluate`` / ``inference.run_inference``;
- the ``mars-rec predict`` / ``evaluate`` / ``visualize`` CLI commands.

Training runs **once per module** (``trained_run``); every test reads the
same task directory. Tests that mutate it (``run_evaluate`` rewrites
``metadata.json``) compare against values captured right after training,
so the order they run in does not matter.

Marked ``slow``: the shared training run takes ~15s on CPU.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.slow

RANKING_KEYS = {f"{m}@{k}" for m in ("ndcg", "hit_rate", "mrr", "map") for k in (5, 10, 20)}


def _build_task(**overrides):
    from mars_rec.simulation.training import SupervisedModelTraining

    defaults = dict(
        project="tests._fixtures.dummy_project.dummy_project_config",
        recommender_module_class="mars_rec.model.base_model.LogisticRegression",
        recommender_extra_params={"n_factors": 4},
        epochs=2,
        batch_size=8,
        loss_function="mse",  # LogisticRegression returns [B, 1]; mse broadcasts
        optimizer="adam",
        learning_rate=1e-3,
        metrics=["loss"],
        ranking_metrics=True,
        device="cpu",
        generator_workers=0,
    )
    defaults.update(overrides)
    return SupervisedModelTraining(**defaults)


@dataclass
class TrainedRun:
    root: Path
    task_dir: Path
    task_id: str
    seed: int
    test_csv: Path
    #: ``metadata.json`` exactly as training wrote it (before any re-eval).
    metadata: dict


def _clear_luigi_instance_cache():
    # Luigi caches Task instances by parameter values; ``run_evaluate``
    # rebuilds the task from params.json, which would otherwise hand back
    # the stateful instance that trained the model.
    import luigi

    luigi.task_register.Register.clear_instance_cache()


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """Train once and expose the task directory to every test.

    - ``OUTPUT_PATH`` defaults to the relative ``output/`` (resolved from
      cwd), so chdir'ing into a tmp dir catches every artifact.
    - ``MARS_TEST_DATASET_DIR`` is read by the fixture project's prep task.
    """
    root = tmp_path_factory.mktemp("lifecycle")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MARS_TEST_DATASET_DIR", str(root / "ds"))
        mp.chdir(root)
        _clear_luigi_instance_cache()

        task = _build_task()
        # Run the dependency manually so a failure surfaces as a real
        # exception instead of Luigi's swallowed worker-fail summary.
        task.requires().run()
        task.run()

        task_dir = Path(task.output().path).resolve()
        with open(task_dir / "metadata.json") as f:
            metadata = json.load(f)
        (test_csv,) = (root / "ds").glob("test_*.csv")

        yield TrainedRun(
            root=root,
            task_dir=task_dir,
            task_id=task.task_id,
            seed=task.seed,
            test_csv=test_csv,
            metadata=metadata,
        )


@pytest.fixture(autouse=True)
def _fresh_luigi_cache():
    _clear_luigi_instance_cache()
    yield


# ---------------------------------------------------------------------------
# Training artifacts
# ---------------------------------------------------------------------------


class TestTrainingArtifacts:
    def test_writes_required_artifacts(self, trained_run):
        """Every artifact downstream consumers depend on is on disk."""
        for name in (
            "params.json",
            "index_mapping.pkl",
            "history.csv",
            "metadata.json",
            "summary.txt",
            "weights.ckpt",
            "test_set_predictions.csv",
            # Architecture SVG is always emitted with trace=True.
            "architecture.svg",
        ):
            assert (trained_run.task_dir / name).exists(), f"missing artifact: {name}"

        # The diagram must contain at least one tensor shape annotation —
        # otherwise the forward-pass trace silently failed.
        svg_content = (trained_run.task_dir / "architecture.svg").read_text()
        assert "[B" in svg_content, "no tensor shape annotations in architecture.svg"

    def test_metadata_json_records_run_info(self, trained_run):
        """``metadata.json`` must include the task_id, seed, and data sizes —
        the fields the CLI ``results`` command reads."""
        meta = trained_run.metadata
        assert meta["run"]["task_id"] == trained_run.task_id
        assert meta["run"]["seed"] == trained_run.seed
        assert "history" in meta
        assert "n_users" in meta["data"]
        assert "n_items" in meta["data"]

    def test_metadata_records_ranking_and_roc_auc(self, trained_run):
        """With ``ranking_metrics=True`` the post-train hook computes
        NDCG/HitRate/MRR/MAP @{5,10,20} and ROC-AUC and persists them."""
        ranking = trained_run.metadata["ranking_metrics"]
        assert set(ranking) == RANKING_KEYS
        for name, value in ranking.items():
            assert 0.0 <= value <= 1.0, name
        # Monotonic in k: a larger cutoff can only find more hits.
        assert ranking["hit_rate@5"] <= ranking["hit_rate@10"] <= ranking["hit_rate@20"]
        assert ranking["ndcg@5"] <= ranking["ndcg@10"] <= ranking["ndcg@20"]
        # One relevant item per row → MRR and MAP coincide.
        for k in (5, 10, 20):
            assert ranking[f"mrr@{k}"] == pytest.approx(ranking[f"map@{k}"])
        # The fixture catalog has 12 items (< 20 candidates), so the ground
        # truth is always inside the top-20.
        assert ranking["hit_rate@20"] == 1.0

        roc = trained_run.metadata["roc_auc_metrics"]
        assert set(roc) == {"roc_auc"}
        assert 0.0 <= roc["roc_auc"] <= 1.0

    def test_test_set_predictions_are_ranked_per_row(self, trained_run):
        test_df = pd.read_csv(trained_run.test_csv)
        preds = pd.read_csv(trained_run.task_dir / "test_set_predictions.csv")
        assert len(preds) == len(test_df)
        for column in ("sorted_actions", "action_scores", "trained", "item_indexed"):
            assert column in preds.columns
        for raw in preds["action_scores"]:
            scores = json.loads(raw)
            assert scores == sorted(scores, reverse=True)

    def test_index_mapping_persisted_with_pad_oov_reservations(self, trained_run):
        """The saved index_mapping must honour OOV=0 / NaN=1 / -1=2
        reservations from ``create_index_mapping`` — every downstream
        inference call relies on these slots being free."""
        with open(trained_run.task_dir / "index_mapping.pkl", "rb") as f:
            idx = pickle.load(f)

        # np.nan-as-dict-key cannot be looked up directly (nan != nan), so
        # check it via iteration.
        assert idx["item_id"].get("-1") == 2
        assert idx["item_id"].get(None) == 1
        nan_values = [v for k, v in idx["item_id"].items() if isinstance(k, float) and math.isnan(k)]
        assert nan_values == [1], f"expected single nan→1 entry, got {nan_values}"

        # Real items in the fixture all start with "i" — they must NOT
        # collide with the reserved slots (0=OOV, 1=NaN, 2=-1/PAD).
        real_values = [v for k, v in idx["item_id"].items() if isinstance(k, str) and k.startswith("i")]
        assert real_values, "no real items in index_mapping — fixture broken"
        assert all(v >= 3 for v in real_values), f"real items collided with reserved indices: {sorted(real_values)[:5]}"


# ---------------------------------------------------------------------------
# simulation.training.run_evaluate
# ---------------------------------------------------------------------------


class TestRunEvaluateOriginalSplits:
    """Flow 1 — re-evaluate the splits the model was trained on."""

    def test_reproduces_training_time_metrics(self, trained_run):
        """Candidate sampling is seeded per (seed, user, item), so a re-eval
        must give byte-identical ranking metrics — this is what makes
        ``mars-rec results`` comparable across re-evaluations."""
        from mars_rec.simulation.training import run_evaluate

        results = run_evaluate(str(trained_run.task_dir), eval_split="all")

        assert results["ranking_metrics"] == trained_run.metadata["ranking_metrics"]
        assert results["roc_auc_metrics"] == trained_run.metadata["roc_auc_metrics"]
        assert results["val_metrics"]["val_loss"] == pytest.approx(
            trained_run.metadata["evaluation"]["val_loss"], rel=1e-5
        )

    def test_writes_eval_results_and_refreshes_metadata(self, trained_run):
        from mars_rec.simulation.training import run_evaluate

        results = run_evaluate(str(trained_run.task_dir), eval_split="all")

        with open(trained_run.task_dir / "eval_results.json") as f:
            assert json.load(f)["ranking_metrics"] == results["ranking_metrics"]
        with open(trained_run.task_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["ranking_metrics"] == results["ranking_metrics"]
        assert meta["evaluation"] == results["val_metrics"]
        # Fields written by training must survive the refresh.
        assert meta["run"]["task_id"] == trained_run.task_id

    def test_val_split_only_skips_ranking(self, trained_run):
        from mars_rec.simulation.training import run_evaluate

        results = run_evaluate(str(trained_run.task_dir), eval_split="val")
        assert set(results) == {"val_metrics"}
        assert "val_loss" in results["val_metrics"]

    def test_ranking_metrics_override_false(self, trained_run, tmp_path):
        from mars_rec.simulation.training import run_evaluate

        results = run_evaluate(
            str(trained_run.task_dir), eval_split="test", ranking_metrics=False, output_dir=str(tmp_path)
        )
        assert results == {"ranking_metrics": {}, "roc_auc_metrics": {}}

    def test_missing_task_dir_raises(self, tmp_path):
        from mars_rec.simulation.training import run_evaluate

        with pytest.raises(FileNotFoundError):
            run_evaluate(str(tmp_path / "does-not-exist"))


class TestRunEvaluateExternalData:
    """Flow 2 — evaluate an external CSV through the same pipeline."""

    def test_external_copy_of_test_split_matches_training(self, trained_run, tmp_path):
        """Feeding the test split back as "external" data must go through the
        same preprocessing and land on the same ranking metrics."""
        from mars_rec.simulation.training import run_evaluate

        out_dir = tmp_path / "ext"
        metadata_before = (trained_run.task_dir / "metadata.json").read_text()

        results = run_evaluate(
            str(trained_run.task_dir),
            data_path=str(trained_run.test_csv),
            output_dir=str(out_dir),
        )

        assert results["ranking_metrics"] == trained_run.metadata["ranking_metrics"]
        assert results["roc_auc_metrics"] == trained_run.metadata["roc_auc_metrics"]
        assert "val_loss" in results["val_metrics"]
        for name in ("eval_results.json", "test_set_predictions.csv"):
            assert (out_dir / name).exists(), name
        # A separate output_dir must leave the task's metadata.json untouched.
        assert (trained_run.task_dir / "metadata.json").read_text() == metadata_before

    def test_reads_parquet(self, trained_run, tmp_path):
        from mars_rec.simulation.training import run_evaluate

        parquet = tmp_path / "test.parquet"
        pd.read_csv(trained_run.test_csv).to_parquet(parquet)

        results = run_evaluate(str(trained_run.task_dir), data_path=str(parquet), output_dir=str(tmp_path / "out"))
        assert results["ranking_metrics"] == trained_run.metadata["ranking_metrics"]


# ---------------------------------------------------------------------------
# inference.* against a real training run
# ---------------------------------------------------------------------------


class TestInferenceOnTrainedRun:
    def test_inference_run_evaluate_locates_test_split(self, trained_run, tmp_path):
        """Regression: ``_load_test_split`` built ``test_0.2_…`` while the
        prep task writes ``test_0.20_…``, so re-eval mode always raised
        FileNotFoundError with the default ``test_size``."""
        from mars_rec.inference import run_evaluate

        result = run_evaluate(str(trained_run.task_dir), output_path=str(tmp_path / "eval.json"))

        assert result["data_source"] == "test_split"
        assert result["n_samples"] == len(pd.read_csv(trained_run.test_csv))
        assert (tmp_path / "eval.json").exists()

    def test_run_inference_preserves_users_and_row_count(self, trained_run, tmp_path):
        from mars_rec.inference import run_inference

        out = tmp_path / "result.json"
        result = run_inference(str(trained_run.task_dir), str(trained_run.test_csv), str(out), k=5)

        test_df = pd.read_csv(trained_run.test_csv)
        assert result["user"] == test_df["user_id"].tolist()
        assert len(result["top10s"]) == len(test_df)
        with open(out) as f:
            assert json.load(f)["user"] == result["user"]


# ---------------------------------------------------------------------------
# CLI commands against a real training run
# ---------------------------------------------------------------------------


class TestCliOnTrainedRun:
    def _invoke(self, *args):
        from click.testing import CliRunner

        from mars_rec import cli as cli_mod

        result = CliRunner().invoke(cli_mod.cli, list(args), catch_exceptions=False)
        assert result.exit_code == 0, result.output
        return result

    def test_evaluate_prints_summary(self, trained_run, tmp_path):
        result = self._invoke(
            "evaluate",
            "--model-dir",
            str(trained_run.task_dir),
            "--split",
            "test",
            "--ranking-metrics",
            "--output",
            str(tmp_path),
        )
        assert "EVALUATION SUMMARY" in result.output
        assert "ndcg@10" in result.output
        assert "roc_auc" in result.output
        assert (tmp_path / "eval_results.json").exists()

    def test_evaluate_by_task_id(self, trained_run, tmp_path, monkeypatch):
        """``--task-id`` resolves under ``<OUTPUT_PATH>/models/…/results``
        (``OUTPUT_PATH`` is the relative ``output/`` here, hence the chdir)."""
        monkeypatch.chdir(trained_run.root)
        self._invoke("evaluate", "--task-id", trained_run.task_id, "--split", "val", "--output", str(tmp_path))
        assert (tmp_path / "eval_results.json").exists()

    def test_evaluate_external_data_missing_file_exits_nonzero(self, trained_run, tmp_path):
        from click.testing import CliRunner

        from mars_rec import cli as cli_mod

        result = CliRunner().invoke(
            cli_mod.cli, ["evaluate", "--model-dir", str(trained_run.task_dir), "--data", str(tmp_path / "nope.csv")]
        )
        assert result.exit_code == 1
        assert "Data file not found" in result.output

    def test_predict_writes_result_json(self, trained_run, tmp_path):
        out = tmp_path / "result.json"
        self._invoke(
            "predict",
            "--model-dir",
            str(trained_run.task_dir),
            "--data",
            str(trained_run.test_csv),
            "--output",
            str(out),
            "--top-k",
            "3",
        )
        with open(out) as f:
            payload = json.load(f)
        assert set(payload) == {"top10s", "user", "time"}

    def test_visualize_svg_to_custom_path(self, trained_run, tmp_path):
        out = tmp_path / "arch.svg"
        result = self._invoke("visualize", "--model-dir", str(trained_run.task_dir), "--output", str(out))
        assert out.read_text().lstrip().startswith("<")
        assert "SVG diagram saved" in result.output

    def test_visualize_summary_mode(self, trained_run):
        result = self._invoke("visualize", "--model-dir", str(trained_run.task_dir), "--mode", "summary", "--no-trace")
        assert "LogisticRegression" in result.output

    @pytest.mark.parametrize("command", ["predict", "evaluate", "visualize"])
    def test_unknown_model_dir_exits_nonzero(self, command, tmp_path):
        from click.testing import CliRunner

        from mars_rec import cli as cli_mod

        args = [command, "--model-dir", str(tmp_path / "missing")]
        if command == "predict":
            args += ["--data", "x.csv"]
        result = CliRunner().invoke(cli_mod.cli, args)
        assert result.exit_code == 1
        assert "Directory not found" in result.output
