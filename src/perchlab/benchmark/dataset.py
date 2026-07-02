"""Labelled-dataset loading for the benchmark workflow.

Assumption: every audio file inherits the name of its parent folder as its
ground-truth species label. The input directory is therefore either a single
species folder or a folder of species folders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..audio import discover_audio, parse_filename
from ..config import FilenameConfig
from ..errors import BenchmarkError
from ..logging import get_logger

_log = get_logger("benchmark.dataset")


@dataclass(frozen=True)
class LabelledFile:
    """An audio file paired with its ground-truth label."""

    path: Path
    label: str


def load_labelled_dataset(input_dir: Path, filename_config: FilenameConfig) -> list[LabelledFile]:
    """Discover labelled audio files under ``input_dir``.

    Args:
        input_dir: Root directory (a species folder, or a folder of them).
        filename_config: Used only to resolve parent-folder labels consistently.

    Returns:
        Labelled files whose parent folder provides a label.

    Raises:
        BenchmarkError: If no labelled files are found.
    """
    input_dir = Path(input_dir)
    files: list[LabelledFile] = []
    for path in discover_audio(input_dir):
        meta = parse_filename(path, filename_config, input_root=input_dir)
        if meta.expected_label:
            files.append(LabelledFile(path=path, label=meta.expected_label))
    if not files:
        raise BenchmarkError(
            f"No labelled files found under {input_dir}. Expected species subfolders."
        )
    labels = sorted({f.label for f in files})
    _log.info("Loaded %d files across %d labels: %s", len(files), len(labels), labels)
    return files
