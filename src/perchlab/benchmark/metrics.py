"""Classification metrics for the benchmark workflow (scikit-learn based)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from ..logging import get_logger
from .evaluate import EvalData

_log = get_logger("benchmark.metrics")

#: Predicted "label" for windows whose top-1 confidence is below threshold.
NONE_LABEL = "__none__"


def predictions_at_threshold(data: EvalData, threshold: float) -> list[str]:
    """Return per-sample predicted labels, blanking those below ``threshold``."""
    return [
        label if conf >= threshold else NONE_LABEL
        for label, conf in zip(data.top1_label, data.top1_conf, strict=True)
    ]


@dataclass
class ThresholdMetrics:
    """Top-1 metrics computed at a single confidence threshold."""

    threshold: float
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    confusion_labels: list[str] = field(default_factory=list)
    report_text: str = ""

    def scalar_row(self) -> dict[str, float]:
        """Return the scalar metrics as a flat row (for sweep tables/plots)."""
        return {
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_micro": self.precision_micro,
            "recall_micro": self.recall_micro,
            "f1_micro": self.f1_micro,
        }


def compute_threshold_metrics(data: EvalData, threshold: float) -> ThresholdMetrics:
    """Compute top-1 classification metrics at ``threshold``."""
    y_true = data.y_true
    y_pred = predictions_at_threshold(data, threshold)
    labels = data.target_labels

    accuracy = float(accuracy_score(y_true, y_pred))
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_mic, r_mic, f_mic, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    p_pc, r_pc, f_pc, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(p_pc[i]),
            "recall": float(r_pc[i]),
            "f1": float(f_pc[i]),
            "support": int(sup[i]),
        }
        for i, label in enumerate(labels)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    return ThresholdMetrics(
        threshold=threshold,
        accuracy=accuracy,
        precision_macro=float(p_mac),
        recall_macro=float(r_mac),
        f1_macro=float(f_mac),
        precision_micro=float(p_mic),
        recall_micro=float(r_mic),
        f1_micro=float(f_mic),
        per_class=per_class,
        confusion=cm.tolist(),
        confusion_labels=list(labels),
        report_text=str(report),
    )


@dataclass
class CurveMetrics:
    """Threshold-independent ROC / PR metrics (one-vs-rest per class)."""

    roc_auc_macro: float
    pr_auc_macro: float
    per_class_roc_auc: dict[str, float]
    per_class_pr_auc: dict[str, float]
    roc_curves: dict[str, tuple[list[float], list[float]]]
    pr_curves: dict[str, tuple[list[float], list[float]]]


def compute_curve_metrics(data: EvalData) -> CurveMetrics:
    """Compute per-class one-vs-rest ROC and PR curves and their AUCs."""
    y_true = np.asarray(data.y_true)
    roc_auc: dict[str, float] = {}
    pr_auc: dict[str, float] = {}
    roc_curves: dict[str, tuple[list[float], list[float]]] = {}
    pr_curves: dict[str, tuple[list[float], list[float]]] = {}

    for j, label in enumerate(data.target_labels):
        binary = (y_true == label).astype(int)
        scores = data.target_scores[:, j] if data.target_scores.size else np.zeros_like(binary)
        if binary.sum() == 0 or binary.sum() == len(binary):
            # Degenerate: only one class present; AUC undefined.
            continue
        roc_auc[label] = float(roc_auc_score(binary, scores))
        pr_auc[label] = float(average_precision_score(binary, scores))
        fpr, tpr, _ = roc_curve(binary, scores)
        roc_curves[label] = (fpr.tolist(), tpr.tolist())
        prec, rec, _ = precision_recall_curve(binary, scores)
        pr_curves[label] = (rec.tolist(), prec.tolist())

    return CurveMetrics(
        roc_auc_macro=float(np.mean(list(roc_auc.values()))) if roc_auc else float("nan"),
        pr_auc_macro=float(np.mean(list(pr_auc.values()))) if pr_auc else float("nan"),
        per_class_roc_auc=roc_auc,
        per_class_pr_auc=pr_auc,
        roc_curves=roc_curves,
        pr_curves=pr_curves,
    )


def metrics_summary(threshold: ThresholdMetrics, curves: CurveMetrics) -> dict[str, Any]:
    """Bundle scalar metrics into a JSON-serialisable summary."""
    return {
        **threshold.scalar_row(),
        "roc_auc_macro": curves.roc_auc_macro,
        "pr_auc_macro": curves.pr_auc_macro,
        "per_class": threshold.per_class,
        "per_class_roc_auc": curves.per_class_roc_auc,
        "per_class_pr_auc": curves.per_class_pr_auc,
    }
