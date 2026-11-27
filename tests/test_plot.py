"""Smoke tests for ``mars_rec.utils.plot``: every figure renders and saves.

The training task writes these plots into each run's output directory, so a
rendering error would abort the run after training finished.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from mars_rec.utils import plot

RNG = np.random.default_rng(0)
Y_TRUE = RNG.integers(0, 2, size=200)
Y_SCORES = np.clip(Y_TRUE * 0.4 + RNG.random(200) * 0.6, 0, 1)
INFO = dict(model_name="Model", task_id="Task____1____abc", extra_info={"epochs": 3})


def _save(fig, tmp_path, name):
    assert isinstance(fig, Figure)
    path = tmp_path / name
    fig.savefig(path)
    matplotlib.pyplot.close(fig)
    assert path.stat().st_size > 0


def test_plot_history(tmp_path):
    history = pd.DataFrame(
        {"epoch": [0, 1, 2], "loss": [0.9, 0.5, 0.3], "val_loss": [1.0, 0.6, 0.5], "acc": [0.5, 0.7, 0.8]}
    )
    _save(plot.plot_history(history, **INFO), tmp_path, "history.jpg")


@pytest.mark.parametrize(
    "scores",
    [Y_SCORES, np.full(50, 0.5), np.array([0.3]), RNG.random(25_000)],
    ids=["normal", "constant", "single", "large"],
)
def test_plot_scores(tmp_path, scores):
    _save(plot.plot_scores(scores, **INFO), tmp_path, "scores.png")


@pytest.mark.parametrize(
    "y_true,y_scores",
    [(Y_TRUE, Y_SCORES), (np.ones(20, dtype=int), np.full(20, 0.9))],
    ids=["both-classes", "one-class"],
)
def test_plot_confusion_matrix(tmp_path, y_true, y_scores):
    fig = plot.plot_confusion_matrix(y_true, y_scores, threshold=0.5, **INFO)
    ax = fig.axes[0]
    # Count annotations sit at the cell centres of a 2x2 grid, row 0 on top.
    texts = {(round(t.get_position()[0], 2), t.get_text()) for t in ax.texts}
    assert any(x == 0.5 for x, _ in texts) and any(x == 1.5 for x, _ in texts)
    assert [lbl.get_text() for lbl in ax.get_xticklabels()] == ["Negative (0)", "Positive (1)"]
    assert ax.get_ylim() == (2.0, 0.0)
    _save(fig, tmp_path, "confusion.png")


def test_plot_roc_curve(tmp_path):
    _save(plot.plot_roc_curve(Y_TRUE, Y_SCORES, **INFO), tmp_path, "roc.png")


def test_plot_precision_recall_curve(tmp_path):
    _save(plot.plot_precision_recall_curve(Y_TRUE, Y_SCORES, **INFO), tmp_path, "pr.png")


def test_importing_plot_does_not_change_global_matplotlib_style():
    import importlib

    before = dict(matplotlib.rcParams)
    importlib.reload(plot)
    plot.plot_scores(Y_SCORES)
    assert dict(matplotlib.rcParams) == before
