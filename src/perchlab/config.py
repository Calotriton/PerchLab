"""Centralized, typed configuration for PerchLab.

Every meaningful parameter lives here as a :mod:`pydantic` model with a default;
there are no magic numbers scattered through the codebase. Configuration is
resolved with the precedence **CLI > environment > YAML > defaults** by
:func:`load_config`.

The Perch V2 input contract (32 kHz, mono, float32) is encoded in
:class:`PreprocessConfig` and reused by the preprocessing and model layers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .errors import ConfigError

# --- Perch V2 constants (from the official model card) --------------------- #
PERCH_SAMPLE_RATE_HZ: int = 32_000
PERCH_WINDOW_S: float = 5.0
PERCH_EMBEDDING_DIM: int = 1536
DEFAULT_MODEL_NAME: str = "PERCH_V2"

# Environment-variable prefix for overrides, e.g. PERCHLAB_SEED=0.
ENV_PREFIX = "PERCHLAB_"


class ModelConfig(BaseModel):
    """How to load and run the Perch model."""

    name: str = DEFAULT_MODEL_NAME
    #: ``auto`` lets Perch Hoplite pick GPU/CPU; explicit values map to zoo presets.
    backend: Literal["auto", "cpu", "gpu", "onnx"] = "auto"
    embedding_dim: int = PERCH_EMBEDDING_DIM
    sample_rate_hz: int = PERCH_SAMPLE_RATE_HZ
    #: Windows per inference batch. Capped conservatively for small GPUs.
    batch_size: int = 16
    #: Logit->confidence mapping. Perch V2's classifier is softmax-trained (per
    #: the model paper), so ``softmax`` is the default; ``sigmoid`` gives an
    #: independent per-class (multi-label) score but saturates on V2's logits.
    activation: Literal["softmax", "sigmoid"] = "softmax"


class PreprocessConfig(BaseModel):
    """Perch V2 input contract applied to every file before inference."""

    target_sample_rate_hz: int = PERCH_SAMPLE_RATE_HZ
    mono: bool = True
    #: Peak-normalization target amplitude; ``None`` defers to the model default.
    peak_norm: float | None = None
    #: Files shorter than one window (in seconds) are padded to this length.
    min_length_s: float = PERCH_WINDOW_S
    #: Read audio in chunks of this many seconds to bound memory on long files.
    chunk_length_s: float = 600.0


class FilenameConfig(BaseModel):
    """Configurable recorder-filename -> date/time parser.

    The default matches ``PIC02_20250530_040000.wav`` (device, date, time).
    Override ``pattern`` / ``datetime_format`` for AudioMoth, Song Meter, etc.
    """

    #: Regex with named groups ``device``, ``date``, ``time`` (all optional).
    pattern: str = r"(?P<device>[A-Za-z0-9]+)_(?P<date>\d{8})_(?P<time>\d{6})"
    date_format: str = "%Y%m%d"
    time_format: str = "%H%M%S"


class SegmentConfig(BaseModel):
    """Audio-segment extraction for the identification workflow."""

    enabled: bool = False
    output_dir: Path | None = None
    bin_width: float = 0.1
    max_per_bin: int = 20
    clip_duration_s: float = PERCH_WINDOW_S
    #: Extra seconds of surrounding audio added to *each* side of the central
    #: clip for context; the written clip is ``clip_duration_s + 2 * context_s``
    #: long (0 = no padding).
    context_s: float = 0.0
    seed: int | None = None

    @model_validator(mode="after")
    def _check_context(self) -> SegmentConfig:
        if self.context_s < 0:
            raise ConfigError("segments.context_s must be >= 0.")
        return self


class ThresholdSweep(BaseModel):
    """A confidence-threshold sweep shared by identification and benchmark."""

    enabled: bool = False
    start: float = 0.1
    end: float = 1.0
    step: float = 0.1

    @model_validator(mode="after")
    def _check_bounds(self) -> ThresholdSweep:
        if self.enabled:
            if not (0.0 <= self.start <= 1.0 and 0.0 <= self.end <= 1.0):
                raise ConfigError("Sweep thresholds must be within [0, 1].")
            if self.step <= 0:
                raise ConfigError("Sweep step must be positive.")
            if self.end < self.start:
                raise ConfigError("Sweep end must be >= start.")
        return self

    def values(self) -> list[float]:
        """Return the discrete thresholds in the sweep (inclusive of ``end``)."""
        out: list[float] = []
        v = self.start
        # Guard against float drift by rounding to a sane precision.
        while v <= self.end + 1e-9:
            out.append(round(v, 6))
            v += self.step
        return out


class IdentifyConfig(BaseModel):
    """Parameters for Workflow 1 - Species Identification."""

    input_dir: Path | None = None
    output_dir: Path | None = None
    window_s: float = PERCH_WINDOW_S
    hop_s: float = PERCH_WINDOW_S
    top_k: int = 3
    #: With softmax scoring, co-occurring species split the probability mass, so
    #: a low default keeps those detections (matches typical Perch usage).
    threshold: float = 0.1
    sweep: ThresholdSweep = Field(default_factory=ThresholdSweep)
    segments: SegmentConfig = Field(default_factory=SegmentConfig)
    #: Output formats to write: any of {"csv", "parquet", "raven"}.
    formats: list[str] = Field(default_factory=lambda: ["csv", "raven"])


class EmbedConfig(BaseModel):
    """Parameters for Workflow 2 - Embedding Generation."""

    input_dir: Path | None = None
    output_dir: Path | None = None
    window_s: float = PERCH_WINDOW_S
    hop_s: float = PERCH_WINDOW_S
    embedding_dim: int = PERCH_EMBEDDING_DIM
    labeled: bool = False
    #: Optional portable export in addition to the Hoplite DB.
    export: Literal["none", "parquet", "npz"] = "none"


class BenchmarkConfig(BaseModel):
    """Parameters for Workflow 3 - Benchmark."""

    input_dir: Path | None = None
    output_dir: Path | None = None
    window_s: float = PERCH_WINDOW_S
    hop_s: float = PERCH_WINDOW_S
    #: Primary operating point for the full metric report; low to suit softmax
    #: scores (see :class:`ModelConfig.activation`).
    threshold: float = 0.1
    sweep: ThresholdSweep = Field(default_factory=ThresholdSweep)
    #: Evaluation unit; "window" (default) or "file" aggregation.
    aggregate: Literal["window", "file"] = "window"
    mode: Literal["classification", "annotations", "compare"] = "classification"


class OptimalThresholdConfig(BaseModel):
    """Parameters for Workflow 4 - Optimal Confidence Threshold Detection.

    Fits, per species, a logistic regression of *correct vs. incorrect* on the
    logit-transformed confidence score of human-validated detections, then solves
    for the confidence at which a detection has :attr:`target_probability` of
    being correct. See the README for the full methodology.
    """

    input_dir: Path | None = None
    output_dir: Path | None = None
    window_s: float = PERCH_WINDOW_S
    hop_s: float = PERCH_WINDOW_S
    #: Target scientific name (Perch class). ``None`` estimates a separate
    #: threshold for every species folder found under ``input_dir``.
    species: str | None = None
    #: Probability of correct identification the threshold is solved for.
    target_probability: float = 0.95
    #: Lower edges of the confidence-score categories in the precision table
    #: (each bin runs to the next edge; the last runs to 1.0).
    bin_edges: list[float] = Field(default_factory=lambda: [0.1, 0.3, 0.5])

    @model_validator(mode="after")
    def _check(self) -> OptimalThresholdConfig:
        if not (0.0 < self.target_probability < 1.0):
            raise ConfigError("target_probability must be in (0, 1).")
        if not self.bin_edges or any(not (0.0 <= e <= 1.0) for e in self.bin_edges):
            raise ConfigError("bin_edges must be non-empty and within [0, 1].")
        if list(self.bin_edges) != sorted(self.bin_edges):
            raise ConfigError("bin_edges must be ascending.")
        return self


class AppConfig(BaseModel):
    """Top-level PerchLab configuration."""

    seed: int | None = 0
    log_level: str = "INFO"
    log_file: Path | None = None
    model: ModelConfig = Field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    filename: FilenameConfig = Field(default_factory=FilenameConfig)
    identify: IdentifyConfig = Field(default_factory=IdentifyConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    optimal_threshold: OptimalThresholdConfig = Field(default_factory=OptimalThresholdConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_overrides() -> dict[str, Any]:
    """Collect ``PERCHLAB_<SECTION>__<KEY>`` environment overrides.

    Example: ``PERCHLAB_MODEL__BACKEND=onnx`` -> ``{"model": {"backend": "onnx"}}``.
    """
    overrides: dict[str, Any] = {}
    for env_key, raw in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        path = env_key[len(ENV_PREFIX) :].lower().split("__")
        cursor = overrides
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = yaml.safe_load(raw)
    return overrides


def load_config(
    config_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Build an :class:`AppConfig` using CLI > env > YAML > defaults precedence.

    Args:
        config_path: Optional YAML file with a subset of the config tree.
        overrides: Highest-priority values (typically parsed CLI arguments),
            already shaped like the config tree; ``None`` values are ignored.

    Returns:
        A validated :class:`AppConfig`.

    Raises:
        ConfigError: If the YAML file is missing or cannot be parsed.
    """
    data: dict[str, Any] = {}

    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - passthrough
            raise ConfigError(f"Could not parse {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"Config root must be a mapping: {path}")
        data = _deep_merge(data, loaded)

    data = _deep_merge(data, _env_overrides())

    if overrides:
        data = _deep_merge(data, overrides)

    try:
        return AppConfig.model_validate(data)
    except Exception as exc:  # pydantic ValidationError -> ConfigError
        raise ConfigError(str(exc)) from exc
