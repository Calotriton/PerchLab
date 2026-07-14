"""Windowed inference engine shared by all workflows.

Frames a recording into analysis windows (via :class:`AudioPreprocessor`), runs
them through the model in batches, and yields per-window embeddings and logits.
Identification, embedding generation, and benchmarking all build on this single
path so inference is written once.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .logging import get_logger
from .models import PerchModel
from .preprocess import AudioPreprocessor, Window

_log = get_logger("inference")


@dataclass(frozen=True)
class WindowResult:
    """Model outputs for a single analysis window.

    Attributes:
        start_s: Window start offset in seconds.
        end_s: Window end offset in seconds.
        embedding: Pooled embedding vector ``[embedding_dim]``.
        logits: Per-class logits ``[num_classes]``.
    """

    start_s: float
    end_s: float
    embedding: np.ndarray
    logits: np.ndarray


class InferenceEngine:
    """Run a Perch model over a recording's windows in batches."""

    def __init__(
        self,
        model: PerchModel,
        preprocessor: AudioPreprocessor,
        *,
        window_s: float,
        hop_s: float,
        batch_size: int,
    ) -> None:
        """Initialise the engine.

        Args:
            model: The loaded :class:`PerchModel`.
            preprocessor: Configured :class:`AudioPreprocessor`.
            window_s: Analysis window length in seconds.
            hop_s: Hop between windows in seconds.
            batch_size: Number of windows per model batch (memory/throughput knob).
        """
        self.model = model
        self.preprocessor = preprocessor
        self.window_s = window_s
        self.hop_s = hop_s
        self.batch_size = max(1, batch_size)
        # Apply the configured per-window normalization to the model. ``hoplite``
        # leaves the model's native ``target_peak`` untouched; ``none`` disables
        # it so raw audio reaches the network (see PreprocessConfig.normalize).
        if preprocessor.config.normalize == "none":
            model.set_target_peak(None)

    def run_file(self, path: Path) -> Iterator[WindowResult]:
        """Yield :class:`WindowResult` for every window of ``path``.

        Raises:
            AudioError: Propagated from the preprocessor if the file is unreadable.
        """
        batch: list[Window] = []
        for window in self.preprocessor.iter_windows(path, self.window_s, self.hop_s):
            batch.append(window)
            if len(batch) >= self.batch_size:
                yield from self._flush(batch)
                batch = []
        if batch:
            yield from self._flush(batch)

    def _flush(self, batch: list[Window]) -> Iterator[WindowResult]:
        """Run one batch of windows through the model and yield results."""
        audio = np.stack([w.waveform for w in batch], axis=0)
        outputs = self.model.model.batch_embed(audio)
        embeddings = _pool_batched(np.asarray(outputs.embeddings, dtype=np.float32))
        logits = _pool_batched(np.asarray(outputs.logits[self.model.logits_key], dtype=np.float32))
        for i, window in enumerate(batch):
            yield WindowResult(
                start_s=window.start_s,
                end_s=window.end_s,
                embedding=embeddings[i],
                logits=logits[i],
            )


def _pool_batched(array: np.ndarray) -> np.ndarray:
    """Pool a batched model output down to ``[batch, features]``.

    Handles ``[B, frames, channels, features]`` (embeddings) and
    ``[B, frames, classes]`` (logits) by mean-pooling every axis between the
    batch and the final feature axis.
    """
    if array.ndim <= 2:
        return array
    axes = tuple(range(1, array.ndim - 1))
    return array.mean(axis=axes)
