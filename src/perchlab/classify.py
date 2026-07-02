"""Turn per-window logits into ranked, thresholded species detections.

Design note (efficiency): inference runs **once** per file. For each window we
cache the fixed top-k predictions with their confidences
(:class:`WindowPrediction`). Applying a confidence threshold — including a whole
multi-threshold sweep — is then a cheap filter over that cache, guaranteeing
identical inference results across thresholds.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .audio import RecordingMeta
from .detections import Detection
from .inference import WindowResult
from .taxonomy import TaxonomyMap


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise logistic sigmoid.

    Perch is multi-label, so per-class sigmoid (not softmax) maps each logit to
    an independent confidence in ``[0, 1]``.
    """
    out = np.empty_like(logits, dtype=np.float64)
    pos = logits >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    exp_neg = np.exp(logits[~pos])
    out[~pos] = exp_neg / (1.0 + exp_neg)
    return out


@dataclass(frozen=True)
class RankedPrediction:
    """One ranked class prediction for a window (pre-threshold)."""

    rank: int
    class_index: int
    confidence: float


@dataclass(frozen=True)
class WindowPrediction:
    """Cached top-k predictions for a single window.

    Attributes:
        start_s: Window start offset in seconds.
        end_s: Window end offset in seconds.
        ranked: Top-k predictions ordered by descending confidence.
    """

    start_s: float
    end_s: float
    ranked: list[RankedPrediction]


class ClassifierRunner:
    """Convert :class:`WindowResult` logits into detections."""

    def __init__(self, taxonomy: TaxonomyMap, *, top_k: int) -> None:
        """Initialise the runner.

        Args:
            taxonomy: Class-name mapping for labels/codes.
            top_k: Number of highest-ranked predictions to keep per window.
        """
        self.taxonomy = taxonomy
        self.top_k = max(1, top_k)

    def predict_windows(self, results: Iterable[WindowResult]) -> list[WindowPrediction]:
        """Compute cached top-k predictions for each window (threshold-free).

        Args:
            results: Per-window model outputs from the inference engine.

        Returns:
            One :class:`WindowPrediction` per window.
        """
        predictions: list[WindowPrediction] = []
        for result in results:
            confidences = sigmoid(result.logits)
            top_idx = _top_k_indices(confidences, self.top_k)
            ranked = [
                RankedPrediction(
                    rank=rank + 1, class_index=int(idx), confidence=float(confidences[idx])
                )
                for rank, idx in enumerate(top_idx)
            ]
            predictions.append(
                WindowPrediction(start_s=result.start_s, end_s=result.end_s, ranked=ranked)
            )
        return predictions

    def detections_at_threshold(
        self,
        windows: Iterable[WindowPrediction],
        *,
        recording: RecordingMeta,
        threshold: float,
        window_s: float,
        hop_s: float,
    ) -> list[Detection]:
        """Filter cached predictions by ``threshold`` into :class:`Detection` rows.

        Windows with no prediction at or above ``threshold`` contribute no rows.

        Args:
            windows: Cached window predictions from :meth:`predict_windows`.
            recording: Source recording metadata.
            threshold: Minimum confidence to retain a prediction.
            window_s: Window length used (for the output rows).
            hop_s: Hop length used (for the output rows).

        Returns:
            Detections, one per retained (window, rank).
        """
        detections: list[Detection] = []
        for window in windows:
            for pred in window.ranked:
                if pred.confidence < threshold:
                    continue
                idx = pred.class_index
                detections.append(
                    Detection(
                        recording=recording,
                        start_s=window.start_s,
                        end_s=window.end_s,
                        window_s=window_s,
                        hop_s=hop_s,
                        rank=pred.rank,
                        label=self.taxonomy.label(idx),
                        common_name=self.taxonomy.common_name(idx),
                        species_code=self.taxonomy.species_code(idx),
                        confidence=pred.confidence,
                        threshold=threshold,
                        expected_label=recording.expected_label,
                    )
                )
        return detections


def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the ``k`` largest values, ordered descending."""
    k = min(k, values.shape[0])
    if k <= 0:
        return np.empty(0, dtype=int)
    part = np.argpartition(values, -k)[-k:]
    return part[np.argsort(values[part])[::-1]]
