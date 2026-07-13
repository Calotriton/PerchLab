"""Load a human-validated dataset for threshold estimation.

The workflow needs, per species, a set of detections each tagged *correct*
(species truly present in the window) or *incorrect* (a false positive). That
verdict comes from the folder a clip sits in. Two layouts are accepted:

* **Single species** (pass ``species``)::

      input_dir/
        correct/    (or present/ true/ tp/ 1/ yes)
        incorrect/  (or absent/ false/ fp/ 0/ no)

* **Multiple species** (one folder per scientific name)::

      input_dir/
        Periparus ater/{correct,incorrect}/...
        Certhia brachydactyla/{correct,incorrect}/...

Clips are typically the 5 s detection windows exported by the identification
workflow's ``--extract`` option, then sorted by a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..audio import discover_audio
from ..errors import ThresholdError
from ..logging import get_logger

_log = get_logger("threshold.dataset")

#: Case-insensitive folder names that mark a clip as a correct / incorrect verdict.
CORRECT_NAMES = {"correct", "present", "true", "tp", "1", "yes", "positive"}
INCORRECT_NAMES = {"incorrect", "absent", "false", "fp", "0", "no", "negative"}


@dataclass(frozen=True)
class ValidatedFile:
    """A validated clip: its path, target species, and correctness verdict."""

    path: Path
    species: str
    correct: bool


def _verdict(name: str) -> bool | None:
    """Map a folder name to ``True``/``False``/``None`` (not a verdict folder)."""
    key = name.strip().lower()
    if key in CORRECT_NAMES:
        return True
    if key in INCORRECT_NAMES:
        return False
    return None


def _has_verdict_subdirs(directory: Path) -> bool:
    """True if ``directory`` contains at least one correct/incorrect subfolder."""
    return any(_verdict(d.name) is not None for d in directory.iterdir() if d.is_dir())


def _collect_species(species_dir: Path, species: str) -> list[ValidatedFile]:
    """Collect validated clips for one species from its verdict subfolders."""
    files: list[ValidatedFile] = []
    for sub in sorted(species_dir.iterdir()):
        if not sub.is_dir():
            continue
        verdict = _verdict(sub.name)
        if verdict is None:
            continue
        for path in discover_audio(sub):
            files.append(ValidatedFile(path=path, species=species, correct=verdict))
    return files


def load_validated_dataset(input_dir: Path, *, species: str | None) -> list[ValidatedFile]:
    """Discover validated clips under ``input_dir``.

    Args:
        input_dir: Dataset root (see module docstring for the two layouts).
        species: Target species for the single-species layout; ``None`` treats
            each top-level folder as a species.

    Returns:
        All validated clips found.

    Raises:
        ThresholdError: If the layout is unusable or no clips are found.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ThresholdError(f"Input folder does not exist: {input_dir}")

    if _has_verdict_subdirs(input_dir):
        # Single-species layout: input_dir holds correct/incorrect directly.
        if not species:
            raise ThresholdError(
                "Found correct/incorrect folders at the top level; pass --species "
                "so the detections can be scored against a target class."
            )
        files = _collect_species(input_dir, species)
    else:
        # Multi-species layout: one folder per species, each with verdict subdirs.
        files = []
        for species_dir in sorted(input_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            name = species if species else species_dir.name
            if not _has_verdict_subdirs(species_dir):
                _log.warning("Skipping '%s': no correct/incorrect subfolders.", species_dir.name)
                continue
            files.extend(_collect_species(species_dir, name))

    if not files:
        raise ThresholdError(
            f"No validated clips found under {input_dir}. Expected correct/ and "
            "incorrect/ subfolders (see 'Optimal confidence threshold' in the README)."
        )
    species_seen = sorted({f.species for f in files})
    _log.info("Loaded %d validated clips across %d species: %s",
              len(files), len(species_seen), species_seen)
    return files
