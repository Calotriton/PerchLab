"""Small cross-cutting helpers: determinism and run manifests."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .logging import get_logger

_log = get_logger("util")


def set_global_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and (if importable) TensorFlow for reproducibility.

    Args:
        seed: The seed value; ``None`` leaves RNGs untouched.
    """
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # TensorFlow is optional; only seed it if already available.
        import tensorflow as tf  # noqa: PLC0415

        tf.random.set_seed(seed)
    except Exception:  # pragma: no cover - TF not installed / not needed
        pass


def timestamp_slug(now: datetime | None = None) -> str:
    """Return a filesystem-safe ``YYYYMMDD_HHMMSS`` timestamp."""
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def _git_sha() -> str | None:
    """Return the current git commit SHA, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - not a git repo / no git
        return None


def write_manifest(
    output_dir: Path,
    *,
    workflow: str,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a ``manifest.json`` capturing the run for reproducibility.

    Records the resolved config, workflow name, timestamp, platform, git SHA,
    and key package versions.

    Args:
        output_dir: Directory to write ``manifest.json`` into (created if needed).
        workflow: Name of the workflow that produced the outputs.
        config: The resolved configuration as a plain dict.
        extra: Optional additional fields to record.

    Returns:
        Path to the written manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    versions: dict[str, str | None] = {}
    for pkg in ("perchlab", "perch-hoplite", "numpy", "pandas", "scikit-learn"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None

    manifest = {
        "workflow": workflow,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_sha": _git_sha(),
        "versions": versions,
        "config": config,
    }
    if extra:
        manifest.update(extra)

    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    _log.debug("Wrote run manifest to %s", path)
    return path
