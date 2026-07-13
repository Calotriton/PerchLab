"""Recover each validated clip's Perch confidence for its target species.

A validated clip is one detection window a human labelled correct/incorrect. To
place it on the confidence axis we run Perch over the clip and read the softmax
(or sigmoid) confidence of the *target species'* class -- the score Perch would
have attached to that detection. For a clip longer than one window we take the
strongest window (the max), i.e. the window that drove the detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..classify import get_activation
from ..errors import AudioError, ThresholdError
from ..inference import InferenceEngine
from ..logging import get_logger
from ..models import PerchModel
from .dataset import ValidatedFile

_log = get_logger("threshold.collect")


@dataclass
class SpeciesScores:
    """Confidences and correctness verdicts collected for one species."""

    species: str
    confidence: list[float] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(confidence, correct)`` as numpy arrays."""
        return (
            np.asarray(self.confidence, dtype=np.float64),
            np.asarray(self.correct, dtype=bool),
        )


def collect_scores(
    files: list[ValidatedFile],
    model: PerchModel,
    engine: InferenceEngine,
    *,
    activation: str = "softmax",
    progress_advance: object = None,
) -> dict[str, SpeciesScores]:
    """Run Perch over ``files`` and gather per-species (confidence, correct) pairs.

    Args:
        files: Validated clips to score.
        model: Loaded model (for its class-name index).
        engine: Configured inference engine.
        activation: Logit->confidence mapping; must match the identification
            workflow so the thresholds transfer to real runs.
        progress_advance: Optional zero-arg callable invoked once per file.

    Returns:
        Species name -> :class:`SpeciesScores`.

    Raises:
        ThresholdError: If a requested species is not one of Perch's classes.
    """
    activation_fn = get_activation(activation)
    class_index = {name: i for i, name in enumerate(model.class_names)}
    scores: dict[str, SpeciesScores] = {}

    for vf in files:
        idx = class_index.get(vf.species)
        if idx is None:
            raise ThresholdError(
                f"Species '{vf.species}' is not a Perch class name; thresholds "
                "are estimated against the model's own (scientific-name) classes."
            )
        try:
            per_window = [activation_fn(r.logits)[idx] for r in engine.run_file(vf.path)]
        except AudioError as exc:
            _log.warning("Skipping %s: %s", vf.path.name, exc)
            if callable(progress_advance):
                progress_advance()
            continue
        if not per_window:
            if callable(progress_advance):
                progress_advance()
            continue
        bucket = scores.setdefault(vf.species, SpeciesScores(species=vf.species))
        bucket.confidence.append(float(np.max(per_window)))
        bucket.correct.append(vf.correct)
        if callable(progress_advance):
            progress_advance()

    return scores
