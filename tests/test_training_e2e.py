"""End-to-end Luigi lifecycle for ``SupervisedModelTraining``.

This is the heavy-weight counterpart to ``test_training_lifecycle.py``:
it actually dispatches the full Luigi task — prep → indexing → training →
evaluation → metadata — against a fixture project under
``tests/_fixtures/dummy_project.py``. Catches regressions in any of:
- index_mapping construction;
- data-frame loading from the prep task's CSVs;
- Lightning trainer integration (callbacks, checkpoint, history.csv);
- post-train hooks (sample_train scores, test predictions, metadata.json,
  architecture diagram).

Marked ``slow`` because a full epoch over the fixture takes ~50s on CPU.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import pytest


pytestmark = pytest.mark.slow


@pytest.fixture
def lifecycle_env(tmp_path, monkeypatch):
    """Wires the framework to write under ``tmp_path``.

    - ``OUTPUT_PATH`` defaults to the relative path ``output/`` (resolved
      from cwd), so chdir'ing into tmp_path catches every artifact.
    - ``MARS_TEST_DATASET_DIR`` is read by the fixture project's prep
      task to decide where to write its CSVs.
    """
    import luigi

    # Luigi caches Task instances by parameter values. Every test here
    # builds a task with identical params, so without clearing the cache
    # later tests receive the *same* instance — with state (lazy-loaded
    # index mapping, data frames) left over from an earlier test whose
    # tmp_path has a different cwd. That made the suite order-dependent.
    luigi.task_register.Register.clear_instance_cache()

    dataset_dir = tmp_path / "ds"
    monkeypatch.setenv("MARS_TEST_DATASET_DIR", str(dataset_dir))
    monkeypatch.chdir(tmp_path)
    return tmp_path


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
        ranking_metrics=False,
        device="cpu",
        generator_workers=0,
    )
    defaults.update(overrides)
    return SupervisedModelTraining(**defaults)


class TestSupervisedTrainingFullLifecycle:
    def test_full_lifecycle_writes_required_artifacts(self, lifecycle_env):
        """Runs the full Luigi pipeline (prep → train → eval) and verifies
        every artifact downstream consumers depend on is on disk."""
        task = _build_task()
        # Run dependency manually so a failure surfaces as a real exception
        # instead of Luigi's swallowed worker-fail summary.
        task.requires().run()
        task.run()

        out = Path(task.output().path)
        # The 6 artifacts ``inference.load_trained_model`` + the registry need.
        for name in (
            "params.json",
            "index_mapping.pkl",
            "history.csv",
            "metadata.json",
            "summary.txt",
            "weights.ckpt",
            # Architecture SVG is always emitted with trace=True (Round 2 viz refactor).
            "architecture.svg",
        ):
            assert (out / name).exists(), f"missing artifact: {name}"

        # The diagram must contain at least one tensor shape annotation —
        # otherwise the forward-pass trace silently failed.
        svg_content = (out / "architecture.svg").read_text()
        assert "[B" in svg_content, "no tensor shape annotations in architecture.svg"

    def test_metadata_json_records_run_info(self, lifecycle_env):
        """``metadata.json`` must include the task_id, seed, and best-epoch
        metrics — the fields the CLI ``results`` command reads."""
        task = _build_task()
        task.requires().run()
        task.run()

        with open(Path(task.output().path) / "metadata.json") as f:
            meta = json.load(f)
        assert meta["run"]["task_id"] == task.task_id
        assert meta["run"]["seed"] == task.seed
        assert "best_epoch" in meta.get("history", {}) or "history" in meta
        assert "n_users" in meta["data"]
        assert "n_items" in meta["data"]

    def test_index_mapping_persisted_with_pad_oov_reservations(self, lifecycle_env):
        """The saved index_mapping must honour OOV=0 / NaN=1 / -1=2
        reservations from ``create_index_mapping`` — every downstream
        inference call relies on these slots being free."""
        task = _build_task()
        task.requires().run()
        task.run()

        with open(Path(task.output().path) / "index_mapping.pkl", "rb") as f:
            idx = pickle.load(f)

        # PAD-string and None both map to reserved slots — assert the values
        # exist as keys with the expected indices. (np.nan-as-dict-key cannot
        # be looked up directly because nan != nan; we check via iteration.)
        import math

        assert idx["item_id"].get("-1") == 2
        assert idx["item_id"].get(None) == 1
        # Iterate keys to confirm a NaN entry exists and maps to 1.
        nan_values = [v for k, v in idx["item_id"].items() if isinstance(k, float) and math.isnan(k)]
        assert nan_values == [1], f"expected single nan→1 entry, got {nan_values}"

        # Real items in the fixture all start with "i" — they must NOT
        # collide with the reserved slots (0=OOV, 1=NaN, 2=-1/PAD).
        real_values = [v for k, v in idx["item_id"].items() if isinstance(k, str) and k.startswith("i")]
        assert real_values, "no real items in index_mapping — fixture broken"
        assert all(v >= 3 for v in real_values), f"real items collided with reserved indices: {sorted(real_values)[:5]}"
