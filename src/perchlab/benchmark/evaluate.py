"""Run Perch over a labelled dataset and collect scores for evaluation.

Produces the per-window arrays every downstream metric needs: the ground-truth
label, the top-1 predicted label (over all model classes), the top-1 confidence,
and the confidence of each *target* class (for one-vs-rest ROC/PR).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..classify import get_activation
from ..inference import InferenceEngine
from ..logging import get_logger
from ..models import PerchModel
from .dataset import LabelledFile

_log = get_logger("benchmark.evaluate")

#: Label used when a window's top-1 prediction is not among the target classes.
OFF_TARGET = "__other__"


@dataclass
class EvalData:
    """Collected evaluation arrays.

    Attributes:
        target_labels: Sorted ground-truth labels present in the dataset.
        y_true: Ground-truth label per sample.
        top1_label: Top-1 predicted label (over all classes) per sample.
        top1_conf: Confidence of the top-1 prediction per sample.
        target_scores: ``[n_samples, n_targets]`` confidence for each target class.
        aggregate: ``"window"`` or ``"file"``.
    """

    target_labels: list[str]
    y_true: list[str]
    top1_label: list[str]
    top1_conf: np.ndarray
    target_scores: np.ndarray
    aggregate: str


def evaluate_dataset(
    files: list[LabelledFile],
    model: PerchModel,
    engine: InferenceEngine,
    *,
    aggregate: str = "window",
    activation: str = "softmax",
) -> EvalData:
    """Run inference over ``files`` and collect scores.

    Args:
        files: Labelled files to evaluate.
        model: The loaded model (for class names).
        engine: Configured inference engine.
        aggregate: ``"window"`` (each window is a sample) or ``"file"`` (mean-pool
            a file's window scores into one sample).
        activation: Logit->confidence mapping (``softmax`` or ``sigmoid``); must
            match the identification workflow so metrics reflect real scores.

    Returns:
        An :class:`EvalData` bundle.
    """
    target_labels = sorted({f.label for f in files})
    target_idx = _resolve_target_indices(target_labels, model.class_names)
    activation_fn = get_activation(activation)

    y_true: list[str] = []
    top1_label: list[str] = []
    top1_conf: list[float] = []
    target_scores: list[np.ndarray] = []

    from ..errors import AudioError  # noqa: PLC0415

    for lf in files:
        try:
            window_confs = [activation_fn(r.logits) for r in engine.run_file(lf.path)]
        except AudioError as exc:
            _log.warning("Skipping %s: %s", lf.path.name, exc)
            continue
        if not window_confs:
            continue
        rows = np.vstack(window_confs)  # [n_windows, n_classes]
        samples = [rows.mean(axis=0, keepdims=True)] if aggregate == "file" else [rows]
        for block in samples:
            for conf in block:
                best = int(np.argmax(conf))
                y_true.append(lf.label)
                pred = model.class_names[best]
                top1_label.append(pred if pred in set(target_labels) else OFF_TARGET)
                top1_conf.append(float(conf[best]))
                scores = [conf[i] if i >= 0 else 0.0 for i in target_idx]
                target_scores.append(np.array(scores, dtype=np.float32))

    scores_matrix = (
        np.vstack(target_scores) if target_scores else np.empty((0, len(target_labels)))
    )
    return EvalData(
        target_labels=target_labels,
        y_true=y_true,
        top1_label=top1_label,
        top1_conf=np.asarray(top1_conf, dtype=np.float32),
        target_scores=scores_matrix,
        aggregate=aggregate,
    )


def _resolve_target_indices(target_labels: list[str], class_names: list[str]) -> list[int]:
    """Map each target label to its model class index (``-1`` if absent)."""
    lookup = {name: i for i, name in enumerate(class_names)}
    indices = []
    for label in target_labels:
        idx = lookup.get(label, -1)
        if idx < 0:
            _log.warning("Label '%s' is not a model class; its ROC/PR will be degenerate.", label)
        indices.append(idx)
    return indices
