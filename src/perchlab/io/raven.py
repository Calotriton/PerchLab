"""Read and write Raven / BirdNET selection tables.

The column schema matches the tables the team already produces so PerchLab's
output is a drop-in replacement:

``Selection  View  Channel  Begin Time (s)  End Time (s)  Low Freq (Hz)
High Freq (Hz)  Common Name  Species Code  Confidence  Begin Path  File Offset (s)``
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ..detections import Detection
from ..logging import get_logger

_log = get_logger("io.raven")

RAVEN_COLUMNS: list[str] = [
    "Selection",
    "View",
    "Channel",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Common Name",
    "Species Code",
    "Confidence",
    "Begin Path",
    "File Offset (s)",
]

# Raven has no frequency bounds from Perch; these defaults match existing tables.
DEFAULT_LOW_FREQ_HZ: int = 0
DEFAULT_HIGH_FREQ_HZ: int = 12_000
DEFAULT_VIEW: str = "Spectrogram 1"
DEFAULT_CHANNEL: int = 1


def detections_to_raven(detections: Sequence[Detection]) -> pd.DataFrame:
    """Build a Raven selection-table DataFrame from detections."""
    rows = []
    for i, det in enumerate(detections, start=1):
        rows.append(
            {
                "Selection": i,
                "View": DEFAULT_VIEW,
                "Channel": DEFAULT_CHANNEL,
                "Begin Time (s)": round(det.start_s, 4),
                "End Time (s)": round(det.end_s, 4),
                "Low Freq (Hz)": DEFAULT_LOW_FREQ_HZ,
                "High Freq (Hz)": DEFAULT_HIGH_FREQ_HZ,
                "Common Name": det.common_name,
                "Species Code": det.species_code,
                "Confidence": round(det.confidence, 4),
                "Begin Path": str(det.recording.path),
                "File Offset (s)": round(det.start_s, 4),
            }
        )
    return pd.DataFrame(rows, columns=RAVEN_COLUMNS)


def write_selection_table(detections: Sequence[Detection], path: Path) -> Path:
    """Write detections as a tab-delimited Raven selection table.

    Args:
        detections: Detections to write.
        path: Output ``.txt`` path (parents created as needed).

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = detections_to_raven(detections)
    df.to_csv(path, sep="\t", index=False)
    _log.debug("Wrote %d selections to %s", len(df), path)
    return path


def read_selection_table(path: Path) -> pd.DataFrame:
    """Read a Raven/BirdNET selection table into a DataFrame.

    Args:
        path: Path to a tab-delimited selection table.

    Returns:
        The table as a DataFrame (original columns preserved).
    """
    return pd.read_csv(path, sep="\t")
