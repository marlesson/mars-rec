"""Matplotlib helpers that render training-time diagnostics.

Every function returns a ``matplotlib.figure.Figure`` so the caller
can either save it (``fig.savefig(...)``) or render inline. The
training task uses :func:`plot_history`, :func:`plot_scores`,
:func:`plot_roc_curve`, :func:`plot_precision_recall_curve` and
:func:`plot_confusion_matrix` to write ``plot_history.jpg``,
``plot_scores.png`` etc. into the run's output directory.
"""

import functools
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Consistent style: a white background with a light grid ("whitegrid") and
# the "muted" palette, applied only while our figures are drawn so importing
# mars_rec never changes the caller's global matplotlib settings.
_PALETTE = ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67"]
_TRAIN_COLOR = _PALETTE[0]  # blue
_VAL_COLOR = _PALETTE[3]  # red/coral
_STYLE = {
    "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": matplotlib.cycler(color=_PALETTE),
    "grid.color": "#CCCCCC",
    "grid.linestyle": "-",
    "font.size": 10.5,
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
}


def _styled(fn):
    """Render ``fn``'s figure under :data:`_STYLE`."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(_STYLE):
            return fn(*args, **kwargs)

    return wrapper


def _despine(ax) -> None:
    """Hide the top and right spines."""
    ax.spines[["top", "right"]].set_visible(False)


def _density_histogram(ax, values: np.ndarray, color, max_kde_points: int = 10_000) -> None:
    """Density histogram plus a Gaussian KDE line (Scott's bandwidth)."""
    ax.hist(values, bins="auto", density=True, color=color, edgecolor="white", linewidth=0.5, alpha=0.6)
    std = values.std()
    if len(values) < 2 or std == 0:
        return  # a KDE is undefined for a single or constant value
    # Subsample for the KDE: its cost is O(n * grid) and the curve is a summary.
    if len(values) > max_kde_points:
        values = np.random.default_rng(0).choice(values, size=max_kde_points, replace=False)
    bandwidth = std * len(values) ** (-1 / 5)
    grid = np.linspace(values.min(), values.max(), 200)
    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z**2).sum(axis=1) / (len(values) * bandwidth * np.sqrt(2 * np.pi))
    ax.plot(grid, density, color=color, linewidth=1.5)


@_styled
def plot_history(
    history_df: pd.DataFrame,
    model_name: Optional[str] = None,
    task_id: Optional[str] = None,
    extra_info: Optional[dict] = None,
) -> Figure:
    """Plot training history curves (train vs validation) for each metric.

    Args:
        history_df: DataFrame with columns epoch, metric, val_metric, ...
        model_name: Model class name for the title.
        task_id: Task ID hash for identification.
        extra_info: Dict of extra metadata to show (e.g. lr, batch_size).
    """
    metrics = [
        col
        for col in history_df.columns
        if col != "epoch" and not col.startswith("val_") and not col.startswith("running_")
    ]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(6 * n_metrics, 4.5),
        squeeze=False,
    )
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        epochs = history_df["epoch"] if "epoch" in history_df else range(len(history_df))

        # Train line
        ax.plot(
            epochs,
            history_df[metric],
            color=_TRAIN_COLOR,
            linewidth=2,
            label="Train",
            marker="o",
            markersize=3,
        )

        # Validation line
        val_col = f"val_{metric}"
        if val_col in history_df.columns:
            ax.plot(
                epochs,
                history_df[val_col],
                color=_VAL_COLOR,
                linewidth=2,
                label="Validation",
                marker="s",
                markersize=3,
            )

            # Annotate best validation value
            valid_vals = history_df[val_col].dropna()
            if valid_vals.empty:
                continue

            if "loss" in metric:
                best_idx = valid_vals.idxmin()
            else:
                best_idx = valid_vals.idxmax()

            best_epoch = int(epochs.iloc[best_idx]) if hasattr(epochs, "iloc") else best_idx
            best_val = history_df[val_col].iloc[best_idx]

            ax.axvline(best_epoch, color=_VAL_COLOR, linestyle="--", alpha=0.4, linewidth=1)
            ax.annotate(
                f"best: {best_val:.4f}\n(epoch {best_epoch})",
                xy=(best_epoch, best_val),
                xytext=(12, -12 if "loss" not in metric else 12),
                textcoords="offset points",
                fontsize=8,
                color=_VAL_COLOR,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=_VAL_COLOR, lw=1),
            )

        # Formatting
        display_name = metric.replace("_", " ").title()
        ax.set_title(display_name, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(display_name, fontsize=10)
        ax.legend(frameon=True, fancybox=True, shadow=False, fontsize=9)
        ax.tick_params(labelsize=9)

        if metric == "loss":
            ax.set_yscale("log")

        _despine(ax)

    # Suptitle with model info
    title_parts = []
    if model_name:
        title_parts.append(model_name)
    if task_id:
        short_hash = task_id.split("____")[-1][:10] if "____" in task_id else task_id[:10]
        title_parts.append(f"[{short_hash}]")

    suptitle = " ".join(title_parts) if title_parts else "Training History"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    # Subtitle with extra info
    if extra_info:
        info_str = "  |  ".join(f"{k}: {v}" for k, v in extra_info.items())
        fig.text(0.5, 0.97, info_str, ha="center", fontsize=9, color="gray", style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


@_styled
def plot_scores(
    scores: np.ndarray,
    model_name: Optional[str] = None,
    task_id: Optional[str] = None,
    extra_info: Optional[dict] = None,
) -> Figure:
    """Plot the distribution of prediction scores on the test set.

    Args:
        scores: Array of prediction scores.
        model_name: Model class name for the title.
        task_id: Task ID hash for identification.
        extra_info: Dict of extra metadata to show.
    """
    scores = np.array(scores)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Histogram + KDE
    _density_histogram(ax, scores.astype(float).ravel(), _TRAIN_COLOR)

    # Stats annotations
    mean_val = np.mean(scores)
    median_val = np.median(scores)
    std_val = np.std(scores)

    ax.axvline(mean_val, color=_VAL_COLOR, linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.4f}")
    ax.axvline(median_val, color=_PALETTE[2], linestyle=":", linewidth=1.5, label=f"Median: {median_val:.4f}")

    # Stats box
    stats_text = (
        f"N = {len(scores):,}\n"
        f"Mean = {mean_val:.4f}\n"
        f"Std = {std_val:.4f}\n"
        f"Min = {scores.min():.4f}\n"
        f"Max = {scores.max():.4f}"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.9)
    ax.text(
        0.97,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        family="monospace",
    )

    # Formatting
    ax.set_xlabel("Prediction Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(frameon=True, fancybox=True, fontsize=9, loc="upper left")
    ax.tick_params(labelsize=9)
    _despine(ax)

    # Title with model info
    title_parts = ["Score Distribution"]
    if model_name:
        title_parts = [model_name, "— Score Distribution"]
    ax.set_title(" ".join(title_parts), fontsize=13, fontweight="bold", pad=10)

    # Subtitle
    subtitle_parts = []
    if task_id:
        short_hash = task_id.split("____")[-1][:10] if "____" in task_id else task_id[:10]
        subtitle_parts.append(f"[{short_hash}]")
    if extra_info:
        subtitle_parts.append("  |  ".join(f"{k}: {v}" for k, v in extra_info.items()))
    if subtitle_parts:
        fig.text(0.5, 0.96, "  ".join(subtitle_parts), ha="center", fontsize=9, color="gray", style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    return fig


def _make_subtitle(task_id: Optional[str], extra_info: Optional[dict]) -> Optional[str]:
    """Build a subtitle string from task_id and extra_info."""
    parts = []
    if task_id:
        short_hash = task_id.split("____")[-1][:10] if "____" in task_id else task_id[:10]
        parts.append(f"[{short_hash}]")
    if extra_info:
        parts.append("  |  ".join(f"{k}: {v}" for k, v in extra_info.items()))
    return "  ".join(parts) if parts else None


@_styled
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    model_name: Optional[str] = None,
    task_id: Optional[str] = None,
    extra_info: Optional[dict] = None,
) -> Figure:
    """Plot a confusion matrix for binary classification.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_scores: Model prediction scores (probabilities after sigmoid).
        threshold: Decision threshold for converting scores to labels.
        model_name: Model class name for the title.
        task_id: Task ID hash for identification.
        extra_info: Dict of extra metadata to show.
    """
    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_scores) >= threshold).astype(int)

    # Force a 2x2 matrix even when one class is absent from the predictions
    # (or labels). Without ``labels=[0, 1]`` sklearn drops missing classes,
    # leaving cm shape (1, 1) and the downstream ``set_xticklabels`` call
    # for 2 labels raises ``FixedLocator (1) does not match labels (2)`` —
    # which kills ``_save_test_set_predictions`` mid-task.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division-by-zero on empty rows
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list("custom", ["#FFFFFF", _TRAIN_COLOR], N=256)

    # Heatmap: cell (i, j) spans [j, j+1] x [i, i+1], row 0 at the top.
    ax.pcolormesh(cm, cmap=cmap, edgecolors="white", linewidth=1.5)
    ax.set_xlim(0, cm.shape[1])
    ax.set_ylim(cm.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(cm.shape[1]) + 0.5)
    ax.set_yticks(np.arange(cm.shape[0]) + 0.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotate cells with count and percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = "white" if pct > 55 else "black"
            ax.text(
                j + 0.5,
                i + 0.38,
                f"{count}",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
                color=text_color,
            )
            ax.text(
                j + 0.5,
                i + 0.65,
                f"({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )

    ax.set_xlabel("Predicted", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual", fontsize=11, labelpad=8)
    ax.set_xticklabels(["Negative (0)", "Positive (1)"], fontsize=10)
    ax.set_yticklabels(["Negative (0)", "Positive (1)"], fontsize=10, rotation=0)
    ax.tick_params(length=0)

    # Metrics summary
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics_text = (
        f"Accuracy:  {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1 Score:  {f1:.4f}\n"
        f"Threshold: {threshold}"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray", alpha=0.9)
    ax.text(
        1.02,
        0.5,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="left",
        bbox=props,
        family="monospace",
    )

    # Title
    title = "Confusion Matrix"
    if model_name:
        title = f"{model_name} — Confusion Matrix"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    subtitle = _make_subtitle(task_id, extra_info)
    if subtitle:
        fig.text(0.42, 0.96, subtitle, ha="center", fontsize=9, color="gray", style="italic")

    fig.tight_layout(rect=[0, 0, 0.82, 0.94])

    return fig


@_styled
def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: Optional[str] = None,
    task_id: Optional[str] = None,
    extra_info: Optional[dict] = None,
) -> Figure:
    """Plot ROC curve with AUC for binary classification.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_scores: Model prediction scores (probabilities).
        model_name: Model class name for the title.
        task_id: Task ID hash for identification.
        extra_info: Dict of extra metadata to show.
    """
    y_true = np.array(y_true).astype(int)
    y_scores = np.array(y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # ROC curve
    ax.plot(
        fpr,
        tpr,
        color=_TRAIN_COLOR,
        linewidth=2.5,
        label=f"ROC Curve (AUC = {roc_auc:.4f})",
    )

    # Fill under curve
    ax.fill_between(fpr, tpr, alpha=0.15, color=_TRAIN_COLOR)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random (AUC = 0.5)")

    # Find and mark optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]

    ax.plot(best_fpr, best_tpr, "o", color=_VAL_COLOR, markersize=10, zorder=5)
    ax.annotate(
        f"Best threshold: {best_threshold:.3f}\nTPR: {best_tpr:.3f}, FPR: {best_fpr:.3f}",
        xy=(best_fpr, best_tpr),
        xytext=(best_fpr + 0.15, best_tpr - 0.15),
        fontsize=9,
        color=_VAL_COLOR,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=_VAL_COLOR, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    # Formatting
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(frameon=True, fancybox=True, fontsize=10, loc="lower right")
    ax.tick_params(labelsize=9)
    ax.set_aspect("equal")
    _despine(ax)

    # AUC box
    auc_text = f"AUC = {roc_auc:.4f}"
    ax.text(
        0.97,
        0.03,
        auc_text,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=_TRAIN_COLOR,
        ha="right",
        va="bottom",
    )

    # Title
    title = "ROC Curve"
    if model_name:
        title = f"{model_name} — ROC Curve"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    subtitle = _make_subtitle(task_id, extra_info)
    if subtitle:
        fig.text(0.5, 0.96, subtitle, ha="center", fontsize=9, color="gray", style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    return fig


@_styled
def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: Optional[str] = None,
    task_id: Optional[str] = None,
    extra_info: Optional[dict] = None,
) -> Figure:
    """Plot Precision-Recall curve for binary classification.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_scores: Model prediction scores (probabilities).
        model_name: Model class name for the title.
        task_id: Task ID hash for identification.
        extra_info: Dict of extra metadata to show.
    """
    y_true = np.array(y_true).astype(int)
    y_scores = np.array(y_scores)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # Baseline = proportion of positives
    baseline = y_true.sum() / len(y_true)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    # PR curve
    ax.plot(
        recall,
        precision,
        color=_TRAIN_COLOR,
        linewidth=2.5,
        label=f"PR Curve (AP = {ap:.4f})",
    )

    # Fill under curve
    ax.fill_between(recall, precision, alpha=0.15, color=_TRAIN_COLOR)

    # Baseline reference
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.3f})")

    # Find and mark best F1 threshold
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]

    ax.plot(best_recall, best_precision, "o", color=_VAL_COLOR, markersize=10, zorder=5)
    ax.annotate(
        f"Best F1: {best_f1:.3f}\nThreshold: {best_threshold:.3f}\nP: {best_precision:.3f}, R: {best_recall:.3f}",
        xy=(best_recall, best_precision),
        xytext=(best_recall - 0.25, best_precision - 0.18),
        fontsize=9,
        color=_VAL_COLOR,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=_VAL_COLOR, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )

    # Formatting
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])
    ax.legend(frameon=True, fancybox=True, fontsize=10, loc="upper right")
    ax.tick_params(labelsize=9)
    _despine(ax)

    # AP box
    ax.text(
        0.03,
        0.03,
        f"AP = {ap:.4f}",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=_TRAIN_COLOR,
        ha="left",
        va="bottom",
    )

    # Title
    title = "Precision-Recall Curve"
    if model_name:
        title = f"{model_name} — Precision-Recall Curve"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    subtitle = _make_subtitle(task_id, extra_info)
    if subtitle:
        fig.text(0.5, 0.96, subtitle, ha="center", fontsize=9, color="gray", style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    return fig
