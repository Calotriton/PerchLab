"""Audio-file discovery and recorder-filename parsing.

Discovery is recursive and format-agnostic (anything libsndfile/ffmpeg can read).
Filenames are parsed into :class:`RecordingMeta` using a *configurable* regex so
PIC, AudioMoth, Song Meter, and other conventions all work (see
:class:`~perchlab.config.FilenameConfig`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime
from datetime import time as time_cls
from pathlib import Path

from .config import FilenameConfig
from .logging import get_logger

_log = get_logger("audio")

#: Extensions we attempt to read. libsndfile/ffmpeg handle the decoding.
AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}
)


@dataclass(frozen=True)
class RecordingMeta:
    """Metadata parsed from a recording's path.

    Attributes:
        path: Absolute path to the audio file.
        device: Recorder identifier parsed from the filename, if any.
        start_date: Recording start date parsed from the filename, if any.
        start_time: Recording start clock time parsed from the filename, if any.
        expected_label: Parent-folder name, used as the ground-truth species
            label in labelled datasets (``None`` when the file sits directly in
            the input root).
    """

    path: Path
    device: str | None
    start_date: date_cls | None
    start_time: time_cls | None
    expected_label: str | None

    @property
    def start_datetime(self) -> datetime | None:
        """Combined start datetime, or ``None`` if date/time were not parsed."""
        if self.start_date is not None and self.start_time is not None:
            return datetime.combine(self.start_date, self.start_time)
        return None

    @property
    def date_str(self) -> str:
        """ISO date string (``YYYY-MM-DD``) or empty string."""
        return self.start_date.isoformat() if self.start_date else ""

    @property
    def time_str(self) -> str:
        """Clock-time string (``HH:MM:SS``) or empty string."""
        return self.start_time.isoformat() if self.start_time else ""


def discover_audio(root: Path, *, recursive: bool = True) -> list[Path]:
    """Return sorted audio files under ``root``.

    Args:
        root: A directory to search, or a single audio file.
        recursive: Whether to descend into subdirectories.

    Returns:
        Sorted list of audio file paths.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() in AUDIO_EXTENSIONS else []

    globber = root.rglob if recursive else root.glob
    files = sorted(
        p for p in globber("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    _log.info("Discovered %d audio file(s) under %s", len(files), root)
    return files


def parse_filename(
    path: Path,
    config: FilenameConfig,
    *,
    input_root: Path | None = None,
) -> RecordingMeta:
    """Parse recorder metadata from a file path.

    Args:
        path: The audio file path.
        config: Filename-pattern configuration.
        input_root: When given, the file's parent folder is used as
            ``expected_label`` only if it differs from ``input_root``.

    Returns:
        A :class:`RecordingMeta`. Missing groups yield ``None`` fields rather
        than raising, so unrecognised names still process (just without dates).
    """
    path = Path(path)
    device = start_date = start_time = None

    match = re.search(config.pattern, path.stem)
    if match:
        groups = match.groupdict()
        device = groups.get("device")
        if groups.get("date"):
            try:
                start_date = datetime.strptime(groups["date"], config.date_format).date()
            except ValueError:
                _log.warning("Could not parse date from %s", path.name)
        if groups.get("time"):
            try:
                start_time = datetime.strptime(groups["time"], config.time_format).time()
            except ValueError:
                _log.warning("Could not parse time from %s", path.name)
    else:
        _log.debug("Filename %s did not match pattern", path.name)

    expected_label = _parent_label(path, input_root)
    return RecordingMeta(
        path=path.resolve(),
        device=device,
        start_date=start_date,
        start_time=start_time,
        expected_label=expected_label,
    )


def _parent_label(path: Path, input_root: Path | None) -> str | None:
    """Return the parent-folder label unless it is the input root itself."""
    parent = path.parent
    if input_root is not None:
        try:
            if parent.resolve() == Path(input_root).resolve():
                return None
        except OSError:  # pragma: no cover - resolution edge cases
            pass
    return parent.name or None


def is_completed(output_path: Path) -> bool:
    """Return ``True`` if ``output_path`` already exists and is non-empty.

    Used for resume/skip-completed behaviour in batch workflows so an
    interrupted run can be restarted without recomputing finished files.
    """
    return output_path.exists() and output_path.stat().st_size > 0
