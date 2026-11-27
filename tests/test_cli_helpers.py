"""Tests for ``mars_rec.cli`` helpers and listing/results subcommands.

Complements ``test_cli.py`` (which covers help/error paths) by exercising
the actual read-side commands ``list`` and ``results``, plus the
``_load_run`` / ``_load_all_runs`` helpers that they both rely on.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from mars_rec import cli as cli_mod


def _make_run_dir(parent: Path, task_id: str, *,
                  project: str = "tests._fixtures.dummy_project",
                  model_cls: str = "models.SomeModel",
                  loss: str = "bce", lr: float = 1e-3,
                  history_rows: int = 3,
                  ranking_metric_val: float | None = None) -> Path:
    """Stage a fake training-run directory with params.json + history.csv +
    optional metadata.json. Mirrors the on-disk shape ``SupervisedModelTraining``
    writes."""
    run_dir = parent / task_id
    run_dir.mkdir()

    params = {
        "project": project,
        "recommender_module_class": model_cls,
        "loss_function": loss,
        "optimizer": "adam",
        "learning_rate": lr,
        "epochs": 5,
        "batch_size": 32,
        "recommender_extra_params": {},
    }
    (run_dir / "params.json").write_text(json.dumps(params))

    if history_rows > 0:
        df = pd.DataFrame({
            "epoch": list(range(history_rows)),
            "loss": [1.0 / (i + 1) for i in range(history_rows)],
            "val_loss": [0.9 / (i + 1) for i in range(history_rows)],
        })
        df.to_csv(run_dir / "history.csv", index=False)

    if ranking_metric_val is not None:
        meta = {
            "ranking_metrics": {"ndcg@10": ranking_metric_val},
            "roc_auc_metrics": {"roc_auc_macro": 0.75},
        }
        (run_dir / "metadata.json").write_text(json.dumps(meta))

    return run_dir


@pytest.fixture
def fake_results_dir(tmp_path, monkeypatch):
    """Stage a results directory with 3 runs and point ``_get_results_dir`` at it."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _make_run_dir(results_dir, "RunA", project="proj.alpha",
                  loss="bce", ranking_metric_val=0.42)
    _make_run_dir(results_dir, "RunB", project="proj.beta",
                  loss="mse", ranking_metric_val=0.31)
    _make_run_dir(results_dir, "RunC_no_history", project="proj.alpha",
                  loss="bce", history_rows=0)

    monkeypatch.setattr(cli_mod, "_get_results_dir", lambda: str(results_dir))
    return results_dir


# ---------------------------------------------------------------------------
# _load_run
# ---------------------------------------------------------------------------

class TestLoadRun:
    def test_returns_none_when_params_missing(self, tmp_path):
        d = tmp_path / "empty_run"
        d.mkdir()
        assert cli_mod._load_run(str(d)) is None

    def test_parses_params_history_and_metadata(self, tmp_path):
        d = _make_run_dir(
            tmp_path, "T1",
            project="proj.x", model_cls="models.M", loss="bce",
            ranking_metric_val=0.55,
        )
        run = cli_mod._load_run(str(d))
        assert run is not None
        assert run["task_id"] == "T1"
        assert run["params"]["loss_function"] == "bce"
        # Best-metrics merged: history columns + ranking_metrics + roc_auc.
        assert run["best_metrics"]["ndcg@10"] == 0.55
        assert run["best_metrics"]["roc_auc_macro"] == 0.75
        assert run["best_metrics"]["total_epochs"] == 3
        assert isinstance(run["timestamp"], datetime)


# ---------------------------------------------------------------------------
# _load_all_runs + list subcommand
# ---------------------------------------------------------------------------

class TestLoadAllRuns:
    def test_returns_all_runs_sorted_newest_first(self, fake_results_dir):
        runs = cli_mod._load_all_runs()
        # 3 staged.
        assert len(runs) == 3
        # Sorted by timestamp descending — stable enough on filesystem
        # mtime resolution.
        timestamps = [r["timestamp"] for r in runs]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_filters_by_project_substring(self, fake_results_dir):
        runs = cli_mod._load_all_runs(project_filter="alpha")
        assert {r["task_id"] for r in runs} == {"RunA", "RunC_no_history"}


class TestListSubcommand:
    def test_list_shows_runs_and_total(self, fake_results_dir):
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["list"])
        assert result.exit_code == 0
        assert "RunA" in result.output
        assert "RunB" in result.output
        assert "Total: 3" in result.output

    def test_list_empty_directory_prints_helpful_message(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty_results"
        empty.mkdir()
        monkeypatch.setattr(cli_mod, "_get_results_dir", lambda: str(empty))
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["list"])
        assert result.exit_code == 0
        assert "No training runs found" in result.output


# ---------------------------------------------------------------------------
# results subcommand
# ---------------------------------------------------------------------------

class TestResultsSubcommand:
    def test_results_filters_by_metric(self, fake_results_dir):
        """``--metric ndcg@10`` should keep only that metric column."""
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["results", "--metric", "ndcg@10"])
        assert result.exit_code == 0
        # ndcg@10 appears in the table header.
        assert "ndcg@10" in result.output
        # roc_auc_macro should NOT appear (filtered out).
        assert "roc_auc_macro" not in result.output

    def test_results_empty_directory_prints_message(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty_results"
        empty.mkdir()
        monkeypatch.setattr(cli_mod, "_get_results_dir", lambda: str(empty))
        runner = CliRunner()
        result = runner.invoke(cli_mod.cli, ["results"])
        assert result.exit_code == 0
        assert "No training runs found" in result.output
