"""Workflow abstraction shared by every PerchLab task.

A :class:`Workflow` declares a name/CLI command and implements two entry points
that both run the *same* logic:

* :meth:`Workflow.run` — execute with a fully-resolved :class:`AppConfig`.
* :meth:`Workflow.configure_interactive` — prompt the user for the parameters
  this workflow needs and return an updated config.

The CLI resolves a workflow from the registry and calls these; interactive and
non-interactive modes therefore never diverge.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..config import AppConfig, ModelConfig
from ..logging import get_logger
from ..models import PerchModel

_log = get_logger("workflow")


@dataclass
class RunSummary:
    """Accumulates per-run statistics for the final report.

    Attributes:
        workflow: Workflow name.
        processed: Number of files processed successfully.
        skipped: Number of files skipped (e.g. already complete).
        failed: Number of files that raised recoverable errors.
        detections: Total detections/embeddings produced (workflow-specific).
        outputs: Paths of notable output files/directories.
    """

    workflow: str
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    detections: int = 0
    outputs: list[Path] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_s(self) -> float:
        """Seconds since the summary was created."""
        return time.perf_counter() - self._start

    def add_output(self, path: Path) -> None:
        """Record a notable output path."""
        self.outputs.append(path)

    def log_final(self) -> None:
        """Emit the human-readable final summary."""
        _log.info(
            "Done: %d processed, %d skipped, %d failed, %d results in %.1fs.",
            self.processed,
            self.skipped,
            self.failed,
            self.detections,
            self.elapsed_s,
        )
        for out in self.outputs:
            _log.info("  output: %s", out)


class Workflow(ABC):
    """Base class for all workflows."""

    #: Registry key and interactive-menu label.
    name: str
    #: Non-interactive CLI subcommand (e.g. ``id``).
    command: str
    #: One-line description for menus and help.
    description: str

    @abstractmethod
    def run(self, config: AppConfig) -> RunSummary:
        """Execute the workflow with a resolved configuration."""

    @abstractmethod
    def configure_interactive(self, config: AppConfig) -> AppConfig:
        """Prompt for this workflow's parameters and return an updated config."""

    # -- shared helpers ----------------------------------------------------- #
    @staticmethod
    def load_model(model_config: ModelConfig) -> PerchModel:
        """Load the Perch model once for a run."""
        return PerchModel.load(model_config)

    def log_parameters(self, params: dict[str, object]) -> None:
        """Log the resolved parameters at the start of a run."""
        _log.info("Workflow: %s", self.name)
        for key, value in params.items():
            _log.info("  %s: %s", key, value)
