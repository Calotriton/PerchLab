"""The :class:`Detection` record and its derived table columns.

A :class:`Detection` is one species prediction for one analysis window at one
rank. It carries everything the CSV and Raven writers need, including
recorder-derived wall-clock timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .audio import RecordingMeta

# Ordered CSV columns for the identification workflow (see README for meanings).
CSV_COLUMNS: list[str] = [
    "date",
    "hour",
    "timestamp",
    "window_size",
    "hop_size",
    "start",
    "end",
    "time",
    "top_k",
    "expected_label",
    "label",
    "threshold",
    "confidence",
    "file",
]


def _seconds_to_clock(seconds: float) -> str:
    """Format a second offset as ``HH:MM:SS`` (drops fractional seconds)."""
    return str(timedelta(seconds=int(seconds)))


@dataclass(frozen=True)
class Detection:
    """A single ranked species prediction within one window.

    Attributes:
        recording: Metadata for the source recording.
        start_s: Window start offset within the recording, in seconds.
        end_s: Window end offset within the recording, in seconds.
        window_s: Analysis window length used, in seconds.
        hop_s: Hop length used, in seconds.
        rank: 1-based top-k rank of this prediction within its window.
        label: Predicted class label (scientific name).
        common_name: Human-readable name (scientific name in v1).
        species_code: Species code (scientific name in v1).
        confidence: Prediction confidence in ``[0, 1]``.
        threshold: Confidence threshold this detection was retained under.
        expected_label: Ground-truth label from the parent folder, if any.
    """

    recording: RecordingMeta
    start_s: float
    end_s: float
    window_s: float
    hop_s: float
    rank: int
    label: str
    common_name: str
    species_code: str
    confidence: float
    threshold: float
    expected_label: str | None

    @property
    def event_clock_time(self) -> str:
        """Wall-clock time of the event (recording start + offset), or empty."""
        start_dt = self.recording.start_datetime
        if start_dt is None:
            return ""
        return (start_dt + timedelta(seconds=self.start_s)).time().isoformat()

    def to_row(self) -> dict[str, Any]:
        """Return this detection as a CSV row keyed by :data:`CSV_COLUMNS`."""
        return {
            "date": self.recording.date_str,
            "hour": self.recording.time_str,
            "timestamp": _seconds_to_clock(self.start_s),
            "window_size": self.window_s,
            "hop_size": self.hop_s,
            "start": round(self.start_s, 3),
            "end": round(self.end_s, 3),
            "time": self.event_clock_time,
            "top_k": self.rank,
            "expected_label": self.expected_label or "",
            "label": self.label,
            "threshold": self.threshold,
            "confidence": round(self.confidence, 6),
            "file": str(self.recording.path),
        }
