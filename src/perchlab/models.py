"""Thin wrapper around Perch Hoplite's model zoo.

:class:`PerchModel` loads a Perch model via
:func:`perch_hoplite.zoo.model_configs.load_model_by_name` and exposes a small,
stable surface for the rest of PerchLab: window metadata, the class-name list,
and helpers that pool the raw :class:`InferenceOutputs` into per-window
embeddings and logits.

We deliberately reuse Hoplite's framing/normalization (via ``embed``) rather than
reimplementing it. The heavy TF/ONNX runtime is only imported when a model is
actually loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import DEFAULT_MODEL_NAME, ModelConfig
from .errors import ModelError
from .logging import get_logger

if TYPE_CHECKING:  # avoid importing the zoo (and TF) at module import time
    from perch_hoplite.zoo import zoo_interface

_log = get_logger("models")

# Map the configured backend to a Perch Hoplite zoo preset (enum *values*).
_PERCH_BACKEND_PRESETS = {
    "auto": "perch_v2",
    "cpu": "perch_v2_cpu",
    "gpu": "perch_v2_gpu",
    "onnx": "perch_v2_onnx",
}


def _resolve_preset_name(config: ModelConfig) -> str:
    """Resolve the zoo preset name from the model config.

    If ``config.name`` is the default (``PERCH_V2``) the backend selects the
    concrete preset; otherwise ``config.name`` is honoured verbatim (lower-cased)
    so other zoo models (SurfPerch, BirdNET, ...) can be requested later.
    """
    if config.name.upper() == DEFAULT_MODEL_NAME:
        return _PERCH_BACKEND_PRESETS[config.backend]
    return config.name.lower()


def _extract_class_names(
    model: zoo_interface.EmbeddingModel, logits_key: str, num_classes: int
) -> list[str]:
    """Return the class-name list aligned with the ``logits_key`` head.

    ``class_list`` is either a bare ``ClassList`` (PlaceholderModel) or a
    ``dict[str, ClassList]`` (TaxonomyModelTF / Perch V2). Perch V2 ships two
    *parallel* lists of equal length — ``labels`` (iNaturalist scientific names,
    which the rest of PerchLab expects for Raven tables and benchmark matching)
    and ``perch_v2_ebird_classes`` (eBird codes) — both aligned to the single
    ``label`` logits head. Selecting by dict order is therefore unsafe: it can
    silently return eBird codes. :func:`_select_class_list` instead matches the
    logits width and the head name.
    """
    class_list = getattr(model, "class_list", None)
    if class_list is None:
        raise ModelError("Model exposes no class_list; cannot map logits to names.")
    chosen = (
        _select_class_list(class_list, logits_key, num_classes)
        if isinstance(class_list, dict)
        else class_list
    )
    classes = getattr(chosen, "classes", None)
    if classes is None:
        raise ModelError("class_list has no `classes` attribute.")
    return list(classes)


def _select_class_list(class_lists: dict[str, Any], logits_key: str, num_classes: int) -> Any:
    """Pick the class list that maps a logits head to names.

    A valid mapping must be exactly as long as the logits axis, so width filters
    first; among the width-matching lists we prefer the one whose key matches the
    head name (``label`` -> ``labels``, else a prefix match), and only then fall
    back to the first candidate. This makes the choice deterministic instead of
    dependent on class-list insertion order.
    """
    if not class_lists:
        raise ModelError("Model exposes an empty class_list; cannot map logits to names.")
    aligned = {
        key: value
        for key, value in class_lists.items()
        if len(getattr(value, "classes", ())) == num_classes
    }
    candidates = aligned or class_lists
    chosen_key = (
        next((key for key in (logits_key, f"{logits_key}s") if key in candidates), None)
        or next((key for key in candidates if key.startswith(logits_key)), None)
        or next(iter(candidates))
    )
    if len(class_lists) > 1:
        _log.debug(
            "Class lists %s available; using '%s' for logits head '%s'.",
            list(class_lists),
            chosen_key,
            logits_key,
        )
    return candidates[chosen_key]


@dataclass
class PerchModel:
    """A loaded Perch model plus the metadata PerchLab needs.

    Attributes:
        model: The underlying Hoplite :class:`EmbeddingModel`.
        name: Resolved preset name (e.g. ``perch_v2``).
        sample_rate: Model input sample rate in Hz (32 kHz for Perch V2).
        window_size_s: Native analysis window in seconds.
        hop_size_s: Native hop in seconds.
        embedding_dim: Embedding width.
        logits_key: Key into ``InferenceOutputs.logits`` used for classification.
        class_names: Ordered class names aligned with the logits' last axis.
    """

    model: zoo_interface.EmbeddingModel
    name: str
    sample_rate: int
    window_size_s: float
    hop_size_s: float
    embedding_dim: int
    logits_key: str
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return len(self.class_names)

    def set_target_peak(self, target_peak: float | None) -> None:
        """Set the model's per-window peak-normalization target.

        Hoplite normalizes each window to ``target_peak`` inside ``embed`` /
        ``batch_embed``; ``None`` disables it (raw audio reaches the network).
        No-op for models that do not expose ``target_peak`` (e.g. the test
        placeholder), which never normalize.

        Args:
            target_peak: Target peak amplitude, or ``None`` to disable.
        """
        if hasattr(self.model, "target_peak"):
            self.model.target_peak = target_peak
            _log.info(
                "Set model peak-normalization target_peak=%s.",
                "None (disabled)" if target_peak is None else target_peak,
            )

    @classmethod
    def load(cls, config: ModelConfig) -> PerchModel:
        """Load a Perch model described by ``config``.

        Args:
            config: The :class:`~perchlab.config.ModelConfig` to load.

        Returns:
            A ready :class:`PerchModel`.

        Raises:
            ModelError: If the model cannot be loaded or lacks logits.
        """
        preset = _resolve_preset_name(config)
        _log.info("Loading Perch model preset '%s' ...", preset)
        try:
            from perch_hoplite.zoo import model_configs  # noqa: PLC0415

            model = model_configs.load_model_by_name(preset)
        except Exception as exc:  # pragma: no cover - depends on runtime/model files
            raise ModelError(f"Failed to load model '{preset}': {exc}") from exc
        return cls._from_loaded(model, name=preset, embedding_dim=config.embedding_dim)

    @classmethod
    def _from_loaded(
        cls,
        model: zoo_interface.EmbeddingModel,
        *,
        name: str,
        embedding_dim: int,
    ) -> PerchModel:
        """Build a :class:`PerchModel` around an already-instantiated model."""
        sample_rate = int(model.sample_rate)
        window_size_s = float(getattr(model, "window_size_s", 5.0))
        hop_size_s = float(getattr(model, "hop_size_s", window_size_s))

        # Probe the logits keys with a single silent window to pick the primary one.
        probe = model.embed(np.zeros(int(sample_rate * window_size_s), dtype=np.float32))
        if not probe.logits:
            raise ModelError(f"Model '{name}' produced no logits; cannot classify.")
        logits_key = "label" if "label" in probe.logits else next(iter(probe.logits))
        num_classes = int(np.asarray(probe.logits[logits_key]).shape[-1])
        class_names = _extract_class_names(model, logits_key, num_classes)

        _log.info(
            "Model '%s' ready: %d classes, window=%.1fs, sr=%d Hz.",
            name,
            len(class_names),
            window_size_s,
            sample_rate,
        )
        return cls(
            model=model,
            name=name,
            sample_rate=sample_rate,
            window_size_s=window_size_s,
            hop_size_s=hop_size_s,
            embedding_dim=embedding_dim,
            logits_key=logits_key,
            class_names=class_names,
        )

    def embed(self, audio: np.ndarray) -> zoo_interface.InferenceOutputs:
        """Run the model on a single mono waveform.

        Args:
            audio: 1-D float32 waveform at :attr:`sample_rate`.

        Returns:
            The raw Hoplite :class:`InferenceOutputs`.
        """
        return self.model.embed(np.asarray(audio, dtype=np.float32))

    def window_embeddings(self, outputs: zoo_interface.InferenceOutputs) -> np.ndarray:
        """Return per-window embeddings ``[frames, embedding_dim]``.

        The channel axis is mean-pooled; the frame (window) axis is preserved so
        each row corresponds to one analysis window.
        """
        emb = np.asarray(outputs.embeddings, dtype=np.float32)
        if emb.ndim == 3:  # [frames, channels, features]
            emb = emb.mean(axis=1)
        return emb

    def window_logits(self, outputs: zoo_interface.InferenceOutputs) -> np.ndarray:
        """Return per-window logits ``[frames, num_classes]``.

        Any channel axis is mean-pooled away.
        """
        logits = np.asarray(outputs.logits[self.logits_key], dtype=np.float32)
        if logits.ndim == 3:  # [frames, channels, classes]
            logits = logits.mean(axis=1)
        return logits
