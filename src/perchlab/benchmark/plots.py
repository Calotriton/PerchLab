"""Matplotlib visualisations for the benchmark workflow.

All functions save a PNG and return its path. Plotting uses the non-interactive
``Agg`` backend so it works headless (WSL/servers).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..logging import get_logger  # noqa: E402
from .metrics import CurveMetrics, ThresholdMetrics  # noqa: E402

_log = get_logger("benchmark.plots")


def plot_confusion_matrix(metrics: ThresholdMetrics, path: Path) -> Path:
    """Save a confusion-matrix heatmap."""
    labels = metrics.confusion_labels
    cm = np.array(metrics.confusion, dtype=float)
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.7), max(4, len(labels) * 0.7)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (threshold={metrics.threshold:.2f})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_roc_curves(curves: CurveMetrics, path: Path) -> Path:
    """Save per-class one-vs-rest ROC curves."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, (fpr, tpr) in curves.roc_curves.items():
        auc = curves.per_class_roc_auc.get(label, float("nan"))
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, path)


def plot_pr_curves(curves: CurveMetrics, path: Path) -> Path:
    """Save per-class one-vs-rest precision-recall curves."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, (rec, prec) in curves.pr_curves.items():
        ap = curves.per_class_pr_auc.get(label, float("nan"))
        ax.plot(rec, prec, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (one-vs-rest)")
    ax.legend(fontsize=7, loc="lower left")
    return _save(fig, path)


def plot_metric_vs_threshold(table: pd.DataFrame, path: Path) -> Path:
    """Save accuracy / precision / recall / F1 versus confidence threshold."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for column, style in (
        ("accuracy", "-o"),
        ("precision_macro", "-s"),
        ("recall_macro", "-^"),
        ("f1_macro", "-d"),
    ):
        ax.plot(table["threshold"], table[column], style, label=column, markersize=4)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Metrics vs confidence threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save and close a figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
