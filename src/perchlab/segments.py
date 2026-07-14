"""Extract audio clips for detections, sampled across confidence bins.

Used by the identification workflow when segment extraction is enabled. Clips
are grouped by confidence into bins, randomly sub-sampled per bin, and written
to per-species subfolders (``Species_subspecies``). Extraction reuses the
already-computed detections and reads audio directly, so no re-inference occurs.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .config import SegmentConfig
from .detections import Detection
from .logging import get_logger

_log = get_logger("segments")


@dataclass
class ExtractionResult:
    """Outcome of a segment-extraction run.

    Attributes:
        clips_written: Number of clips written to disk.
        output_dir: Root directory containing the per-species subfolders.
    """

    clips_written: int
    output_dir: Path


def _sanitize(name: str) -> str:
    """Make a label safe for filenames/folders (``Ardea cinerea`` -> ``Ardea_cinerea``)."""
    return re.sub(r"[^\w.-]+", "_", name.strip()).strip("_") or "unknown"


def _bin_index(confidence: float, bin_width: float) -> int:
    """Return the confidence-bin index for ``confidence``."""
    idx = int(confidence / bin_width)
    n_bins = max(1, int(round(1.0 / bin_width)))
    return min(idx, n_bins - 1)


def extract_segments(
    detections: list[Detection],
    *,
    sample_rate: int,
    config: SegmentConfig,
    output_dir: Path,
) -> ExtractionResult:
    """Extract clips for a sample of detections, organised by species.

    Args:
        detections: Detections to sample from (already thresholded).
        sample_rate: Output clip sample rate in Hz.
        config: Segment-extraction configuration.
        output_dir: Root output directory (created as needed).

    Returns:
        An :class:`ExtractionResult` with the count of clips written.
    """
    from perch_hoplite import audio_io  # noqa: PLC0415

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)

    # Group by confidence bin, then randomly sample within each bin.
    bins: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        bins[_bin_index(det.confidence, config.bin_width)].append(det)

    sampled: list[Detection] = []
    for _bin, items in sorted(bins.items()):
        if len(items) > config.max_per_bin:
            sampled.extend(rng.sample(items, config.max_per_bin))
        else:
            sampled.extend(items)

    written = 0
    for det in sampled:
        try:
            clip = _read_centered_clip(
                audio_io,
                det,
                sample_rate=sample_rate,
                clip_duration_s=config.clip_duration_s,
                context_s=config.context_s,
            )
        except Exception as exc:  # never abort the batch on one bad clip
            _log.warning(
                "Could not extract clip for %s @ %.1fs: %s",
                det.recording.path.name,
                det.start_s,
                exc,
            )
            continue
        species_dir = output_dir / _sanitize(det.label)
        species_dir.mkdir(parents=True, exist_ok=True)
        out_path = species_dir / _clip_filename(det)
        sf.write(out_path, clip, sample_rate)
        written += 1

    _log.info("Extracted %d clips into %s", written, output_dir)
    return ExtractionResult(clips_written=written, output_dir=output_dir)


def _read_centered_clip(
    audio_io: Any,
    det: Detection,
    *,
    sample_rate: int,
    clip_duration_s: float,
    context_s: float = 0.0,
) -> np.ndarray:
    """Read a clip centered on the detection, clamped to bounds.

    The central clip is ``clip_duration_s`` seconds long; ``context_s`` seconds of
    surrounding audio are added on *each* side, so the returned clip spans
    ``clip_duration_s + 2 * context_s`` seconds (e.g. a 5 s clip with 1 s of
    context yields a 7 s clip: 1 s before, the 5 s centre, and 1 s after).
    """
    total_s = clip_duration_s + 2.0 * max(0.0, context_s)
    file_duration = _file_duration_s(det.recording.path)
    center = 0.5 * (det.start_s + det.end_s)
    start = center - total_s / 2.0
    start = max(0.0, min(start, max(0.0, file_duration - total_s)))
    clip = audio_io.load_audio_window(str(det.recording.path), start, sample_rate, total_s)
    clip = np.asarray(clip, dtype=np.float32)
    target = int(round(total_s * sample_rate))
    if clip.size < target:
        clip = np.pad(clip, (0, target - clip.size))
    return clip[:target]


def _file_duration_s(path: Path) -> float:
    """Recording duration in seconds from the file header."""
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _clip_filename(det: Detection) -> str:
    """Build the clip filename per the documented pattern."""
    stem = det.recording.path.stem
    label = _sanitize(det.label)
    parts = [
        stem,
        label,
        f"{det.confidence:.3f}",
        f"{int(det.start_s)}s",
        det.recording.date_str or "NA",
        (det.recording.time_str or "NA").replace(":", ""),
        label,
    ]
    return "_".join(parts) + ".wav"
