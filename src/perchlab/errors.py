"""Exception hierarchy for PerchLab.

All PerchLab-specific errors derive from :class:`PerchLabError` so callers can
catch the whole family with a single ``except``. Per-file failures during batch
processing raise :class:`AudioError` and are logged and skipped rather than
aborting the run (see :mod:`perchlab.preprocess`).
"""

from __future__ import annotations


class PerchLabError(Exception):
    """Base class for all PerchLab errors."""


class ConfigError(PerchLabError):
    """Raised when configuration is missing, malformed, or inconsistent."""


class ModelError(PerchLabError):
    """Raised when the Perch model cannot be loaded or produces unexpected output."""


class AudioError(PerchLabError):
    """Raised when a single audio file cannot be read or preprocessed.

    This is the *recoverable* error: batch workflows catch it per file, log it,
    and continue with the remaining files.
    """


class WorkflowError(PerchLabError):
    """Raised when a workflow is misconfigured or cannot complete."""


class BenchmarkError(PerchLabError):
    """Raised when a benchmark cannot be computed (e.g. no labelled data found)."""


class ThresholdError(PerchLabError):
    """Raised when optimal-threshold estimation cannot proceed (e.g. bad layout)."""
