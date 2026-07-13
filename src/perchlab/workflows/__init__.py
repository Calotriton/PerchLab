"""Workflow registry.

Workflows self-register here so the CLI (interactive menu and subcommands) can
discover them without hard-coded branches. Add a workflow by importing it in
:func:`_register_builtins` and calling :func:`register`; nothing else in the CLI
needs to change.
"""

from __future__ import annotations

from .base import RunSummary, Workflow

_REGISTRY: dict[str, Workflow] = {}


def register(workflow: Workflow) -> None:
    """Register a workflow instance under both its name and CLI command."""
    _REGISTRY[workflow.name] = workflow
    _REGISTRY[workflow.command] = workflow


def get_workflow(key: str) -> Workflow:
    """Look up a workflow by display name or CLI command.

    Raises:
        KeyError: If no workflow matches ``key``.
    """
    if key in _REGISTRY:
        return _REGISTRY[key]
    raise KeyError(f"Unknown workflow: {key}")


def all_workflows() -> list[Workflow]:
    """Return the registered workflows in menu order (deduplicated)."""
    seen: dict[int, Workflow] = {}
    for wf in _REGISTRY.values():
        seen[id(wf)] = wf
    return list(seen.values())


def _register_builtins() -> None:
    """Import and register the built-in workflows."""
    from .benchmark import BenchmarkWorkflow
    from .embed import EmbeddingWorkflow
    from .identify import SpeciesIdentificationWorkflow
    from .threshold import OptimalThresholdWorkflow

    register(SpeciesIdentificationWorkflow())
    register(EmbeddingWorkflow())
    register(BenchmarkWorkflow())
    register(OptimalThresholdWorkflow())


_register_builtins()

__all__ = ["RunSummary", "Workflow", "all_workflows", "get_workflow", "register"]
