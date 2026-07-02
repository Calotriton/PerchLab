"""Interactive prompt helpers built on :mod:`questionary`.

Folder selection uses path autocompletion (not a GUI dialog) so it works over
SSH/WSL and in any terminal. Every prompt has a default so pressing Enter
accepts the configured value.
"""

from __future__ import annotations

from pathlib import Path

import questionary

from .errors import WorkflowError


def _require(value: object) -> object:
    """Abort cleanly if the user cancelled a prompt (Ctrl-C returns ``None``)."""
    if value is None:
        raise WorkflowError("Cancelled.")
    return value


def select(message: str, choices: list[str], *, default: str | None = None) -> str:
    """Ask the user to pick one option from ``choices``."""
    return str(_require(questionary.select(message, choices=choices, default=default).ask()))


def ask_path(
    message: str,
    *,
    default: str | None = None,
    must_exist: bool = True,
) -> Path:
    """Prompt for a filesystem path with tab autocompletion.

    Args:
        message: Prompt text.
        default: Default path shown.
        must_exist: If ``True``, re-prompt until an existing path is given.

    Returns:
        The chosen path.
    """
    while True:
        raw = str(_require(questionary.path(message, default=default or "").ask())).strip()
        path = Path(raw).expanduser()
        if not must_exist or path.exists():
            return path
        questionary.print(f"Path does not exist: {path}", style="fg:red")


def ask_bool(message: str, *, default: bool = False) -> bool:
    """Ask a yes/no question."""
    return bool(_require(questionary.confirm(message, default=default).ask()))


def ask_float(message: str, *, default: float) -> float:
    """Prompt for a float, re-asking on invalid input."""
    while True:
        raw = str(_require(questionary.text(message, default=str(default)).ask())).strip()
        try:
            return float(raw)
        except ValueError:
            questionary.print("Please enter a number.", style="fg:red")


def ask_int(message: str, *, default: int) -> int:
    """Prompt for an integer, re-asking on invalid input."""
    while True:
        raw = str(_require(questionary.text(message, default=str(default)).ask())).strip()
        try:
            return int(raw)
        except ValueError:
            questionary.print("Please enter an integer.", style="fg:red")


def ask_text(message: str, *, default: str = "") -> str:
    """Prompt for free text."""
    return str(_require(questionary.text(message, default=default).ask())).strip()
