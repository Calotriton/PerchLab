"""Perch V2 input-contract preprocessing.

Perch V2 expects **mono, 32 kHz, float32** waveforms. Peak-normalization is
applied by the model itself per analysis window (Hoplite's
``EmbeddingModel.normalize_audio``), so we deliberately do *not* re-implement or
double-apply it here — we reuse the official pipeline and focus on the contract
the model cannot fix for us: channel count, sample rate, dtype, and validity.

Loading and resampling reuse :mod:`perch_hoplite.audio_io`. Long files are read
window-by-window (bounded memory) via :func:`iter_windows`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PreprocessConfig
from .errors import AudioError
from .logging import get_logger

_log = get_logger("preprocess")


@dataclass(frozen=True)
class Window:
    """One analysis window of preprocessed audio.

    Attributes:
        start_s: Window start offset within the recording, in seconds.
        end_s: Window end offset within the recording, in seconds.
        waveform: Mono float32 samples at the target sample rate.
    """

    start_s: float
    end_s: float
    waveform: np.ndarray


class AudioPreprocessor:
    """Load and validate audio against the Perch V2 input contract."""

    def __init__(self, config: PreprocessConfig, sample_rate: int) -> None:
        """Initialise the preprocessor.

        Args:
            config: Preprocessing configuration.
            sample_rate: Target sample rate in Hz (the model's rate).
        """
        self.config = config
        self.sample_rate = sample_rate

    # -- validation --------------------------------------------------------- #
    def validate(self, waveform: np.ndarray) -> None:
        """Assert that ``waveform`` satisfies the Perch input contract.

        Args:
            waveform: The candidate waveform.

        Raises:
            AudioError: If the waveform is not 1-D float32, is empty, or contains
                non-finite samples.
        """
        if waveform.ndim != 1:
            raise AudioError(f"Expected mono (1-D) audio, got shape {waveform.shape}.")
        if waveform.dtype != np.float32:
            raise AudioError(f"Expected float32 audio, got {waveform.dtype}.")
        if waveform.size == 0:
            raise AudioError("Audio is empty.")
        if not np.all(np.isfinite(waveform)):
            raise AudioError("Audio contains NaN or infinite samples.")

    # -- loading ------------------------------------------------------------ #
    def _to_contract(self, audio: np.ndarray) -> np.ndarray:
        """Coerce loaded audio to mono float32 and validate it."""
        audio = np.asarray(audio)
        if audio.ndim == 2:  # (samples, channels) -> mono
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
        if self.config.peak_norm is not None:
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 0:
                audio = (audio / peak * self.config.peak_norm).astype(np.float32)
        self.validate(audio)
        return audio

    def _read_resampled(self, path: Path, offset_s: float, window_s: float) -> np.ndarray:
        """Read audio at the target sample rate using the configured resampler.

        ``window_s < 0`` reads the whole file. ``polyphase`` delegates to Perch
        Hoplite's native loader (``scipy.signal.resample_poly`` via librosa);
        ``soxr_hq`` reads the window and resamples with librosa's soxr filter.
        """
        if self.config.resampler == "soxr_hq":
            import librosa  # noqa: PLC0415

            duration = None if window_s < 0 else window_s
            audio, _ = librosa.load(
                str(path),
                sr=self.sample_rate,
                offset=max(0.0, offset_s),
                duration=duration,
                mono=True,
                res_type="soxr_hq",
            )
            return audio
        from perch_hoplite import audio_io  # noqa: PLC0415

        return audio_io.load_audio_window(str(path), offset_s, self.sample_rate, window_s)

    def load(self, path: Path) -> np.ndarray:
        """Load a full recording as mono/32 kHz/float32.

        Args:
            path: Audio file path.

        Returns:
            The preprocessed waveform.

        Raises:
            AudioError: If the file cannot be read or fails validation.
        """
        try:
            # window<0 reads the whole file, reducing to mono and resampling.
            audio = self._read_resampled(path, 0.0, -1.0)
        except Exception as exc:
            raise AudioError(f"Could not read {path}: {exc}") from exc
        audio = self._to_contract(audio)
        if audio.size < int(self.config.min_length_s * self.sample_rate):
            audio = self._pad(audio, int(self.config.min_length_s * self.sample_rate))
        return audio

    def duration_s(self, path: Path) -> float:
        """Return the recording duration in seconds (cheap header read)."""
        import soundfile as sf  # noqa: PLC0415

        try:
            info = sf.info(str(path))
            return float(info.frames) / float(info.samplerate)
        except Exception as exc:
            raise AudioError(f"Could not read header of {path}: {exc}") from exc

    def iter_windows(
        self,
        path: Path,
        window_s: float,
        hop_s: float,
    ) -> Iterator[Window]:
        """Yield preprocessed analysis windows from a recording.

        Reads window-by-window with :func:`perch_hoplite.audio_io.load_audio_window`
        so memory stays bounded even for multi-hour files. Partial trailing
        windows are zero-padded to ``window_s``.

        Args:
            path: Audio file path.
            window_s: Window length in seconds.
            hop_s: Hop between successive windows in seconds.

        Yields:
            :class:`Window` instances in temporal order.

        Raises:
            AudioError: If the file cannot be read.
        """
        duration = self.duration_s(path)
        window_samples = int(round(window_s * self.sample_rate))
        # Ensure at least one window even for very short files.
        last_start = max(0.0, duration - window_s)
        starts = _frange(0.0, last_start, hop_s)

        for start_s in starts:
            try:
                chunk = self._read_resampled(path, start_s, window_s)
            except Exception as exc:
                raise AudioError(
                    f"Could not read window at {start_s:.1f}s of {path}: {exc}"
                ) from exc
            chunk = self._to_contract(chunk)
            chunk = self._pad(chunk, window_samples)
            yield Window(start_s=start_s, end_s=start_s + window_s, waveform=chunk)

    @staticmethod
    def _pad(audio: np.ndarray, length: int) -> np.ndarray:
        """Right-pad (or truncate) ``audio`` to exactly ``length`` samples."""
        if audio.size == length:
            return audio
        if audio.size > length:
            return audio[:length]
        return np.pad(audio, (0, length - audio.size)).astype(np.float32)


def _frange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range with drift-safe rounding."""
    if step <= 0:
        raise ValueError("step must be positive")
    out: list[float] = []
    v = start
    while v <= stop + 1e-9:
        out.append(round(v, 6))
        v += step
    return out
