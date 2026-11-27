"""Tests for ``mars_rec.cli``.

The CLI is the user entry point — ``mars-rec run``, ``predict``,
``evaluate``, ``visualize``. These tests exercise the Click wiring:
- subcommands are registered;
- required-argument enforcement (``predict`` needs --model-dir OR --task-id);
- luigi dispatch (_run_luigi) and _invalidate_supervised_cache parsing.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from mars_rec import cli as cli_mod


def test_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["--help"])
    assert result.exit_code == 0
    for sub in ("run", "predict", "evaluate", "visualize", "list", "results"):
        assert sub in result.output


def test_run_help_lists_subgroup_commands():
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "supervised" in result.output
    assert "data" in result.output


def test_predict_without_model_dir_or_task_id_exits_nonzero():
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["predict", "--data", "/tmp/no.parquet"])
    assert result.exit_code != 0
    assert "model-dir" in result.output or "task-id" in result.output


def test_predict_with_nonexistent_model_dir_errors_out():
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["predict", "--model-dir", "/tmp/does_not_exist_xyz",
         "--data", "/tmp/no.parquet"],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_evaluate_without_model_dir_or_task_id_exits_nonzero():
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["evaluate"])
    assert result.exit_code != 0
    assert "model-dir" in result.output or "task-id" in result.output


def test_evaluate_with_nonexistent_model_dir_errors_out():
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["evaluate", "--model-dir", "/tmp/does_not_exist_xyz"],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_evaluate_with_nonexistent_data_errors_out(tmp_path):
    # Make a fake model_dir that exists so the data-file check fires.
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["evaluate", "--model-dir", str(model_dir),
         "--data", "/tmp/does_not_exist_data.parquet"],
    )
    assert result.exit_code != 0
    # The CLI prints "Data file not found"
    assert "data" in result.output.lower() and "not found" in result.output.lower()


def test_visualize_without_model_dir_or_task_id_exits_nonzero():
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["visualize"])
    assert result.exit_code != 0


def test_visualize_help_lists_only_svg_and_summary_modes():
    """After the visualization refactor, only ``svg`` and ``summary`` are
    valid modes. ``block`` (graphviz) and ``torchviz`` were removed."""
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["visualize", "--help"])
    assert result.exit_code == 0
    assert "[svg|summary]" in result.output
    assert "block" not in result.output
    assert "torchviz" not in result.output


def test_visualize_rejects_legacy_mode_block():
    """Click should reject ``--mode block`` with the choice list."""
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["visualize", "--model-dir", "/tmp/nope", "--mode", "block"],
    )
    assert result.exit_code != 0
    assert "block" in result.output  # error message lists invalid value


def test_run_luigi_uses_current_interpreter_and_keeps_pythonpath(monkeypatch):
    """Luigi must run under ``sys.executable`` (same venv as the CLI), get
    args as a list (no shell quoting), and keep the caller's PYTHONPATH."""
    import os
    import subprocess
    import sys

    calls = {}

    def fake_run(cmd, env):
        calls["cmd"], calls["env"] = cmd, env
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli_mod.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "src")
    cli_mod._run_luigi(["pkg.mod", "Task", "--obs", "hello world"])

    assert calls["cmd"][:4] == [sys.executable, "-m", "luigi", "--module"]
    assert calls["cmd"][4:] == ["pkg.mod", "Task", "--obs", "hello world", "--local-scheduler"]
    paths = calls["env"]["PYTHONPATH"].split(os.pathsep)
    assert paths[0] == os.getcwd() and "src" in paths


def test_run_luigi_propagates_failure_exit_code(monkeypatch):
    """A failed Luigi run must make the CLI exit non-zero."""
    import subprocess

    monkeypatch.setattr(
        cli_mod.subprocess, "run",
        lambda cmd, env: subprocess.CompletedProcess(cmd, 30),
    )
    with pytest.raises(SystemExit) as exc:
        cli_mod._run_luigi(["pkg.mod", "Task"])
    assert exc.value.code == 30
