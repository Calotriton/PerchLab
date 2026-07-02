"""PerchLab: species identification, embeddings, and benchmarking with Perch V2.

Public API is intentionally small and stable. Heavy submodules (``models``,
``workflows``) are imported lazily by the CLI so that importing :mod:`perchlab`
does not pull in TensorFlow/ONNX runtimes.
"""

from __future__ import annotations

from .config import AppConfig, load_config
from .errors import (
    AudioError,
    BenchmarkError,
    ConfigError,
    ModelError,
    PerchLabError,
    WorkflowError,
)
from .logging import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = [
    "AppConfig",
    "AudioError",
    "BenchmarkError",
    "ConfigError",
    "ModelError",
    "PerchLabError",
    "WorkflowError",
    "__version__",
    "configure_logging",
    "get_logger",
    "load_config",
]
