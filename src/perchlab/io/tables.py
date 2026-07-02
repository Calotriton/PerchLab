"""Tabular detection writers: per-file CSV, Parquet, and run-level summary."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ..detections import CSV_COLUMNS, Detection
from ..logging import get_logger

_log = get_logger("io.tables")


def detections_to_dataframe(detections: Sequence[Detection]) -> pd.DataFrame:
    """Return detections as a DataFrame with the canonical column order."""
    rows = [det.to_row() for det in detections]
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def write_csv(detections: Sequence[Detection], path: Path) -> Path:
    """Write detections to a CSV file (canonical column order).

    Args:
        detections: Detections to write.
        path: Output ``.csv`` path (parents created as needed).

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    detections_to_dataframe(detections).to_csv(path, index=False)
    _log.debug("Wrote %d rows to %s", len(detections), path)
    return path


def write_parquet(detections: Sequence[Detection], path: Path) -> Path:
    """Write detections to a Parquet file (canonical column order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    detections_to_dataframe(detections).to_parquet(path, index=False)
    _log.debug("Wrote %d rows to %s", len(detections), path)
    return path


def write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to CSV or Parquet based on the file suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path
