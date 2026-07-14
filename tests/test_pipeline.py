"""Tests for audio parsing, preprocessing, classification, and IO."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from perchlab.audio import discover_audio, parse_filename
from perchlab.classify import ClassifierRunner, sigmoid, softmax
from perchlab.config import FilenameConfig, PreprocessConfig, SegmentConfig
from perchlab.detections import CSV_COLUMNS, Detection
from perchlab.inference import InferenceEngine, WindowResult
from perchlab.io.raven import RAVEN_COLUMNS, read_selection_table, write_selection_table
from perchlab.io.tables import write_csv
from perchlab.models import PerchModel
from perchlab.preprocess import AudioPreprocessor
from perchlab.segments import extract_segments
from perchlab.taxonomy import TaxonomyMap


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    """A 12-second stereo 48 kHz WAV named like a PIC recorder file."""
    sr = 48_000
    samples = np.random.default_rng(0).normal(0, 0.1, size=(sr * 12, 2)).astype(np.float32)
    path = tmp_path / "PIC02_20250530_040000.wav"
    sf.write(path, samples, sr)
    return path


def test_parse_filename() -> None:
    meta = parse_filename(
        Path("/data/Emberiza schoeniclus/PIC03_20250619_055602.wav"),
        FilenameConfig(),
    )
    assert meta.device == "PIC03"
    assert meta.start_date == date(2025, 6, 19)
    assert meta.start_time == time(5, 56, 2)
    assert meta.expected_label == "Emberiza schoeniclus"


def test_parse_filename_unrecognised() -> None:
    meta = parse_filename(Path("/data/random_clip.wav"), FilenameConfig())
    assert meta.start_date is None and meta.start_time is None  # does not raise


def test_discover_audio(wav_file: Path) -> None:
    found = discover_audio(wav_file.parent)
    assert wav_file in found


def test_preprocess_mono_32k(wav_file: Path) -> None:
    pre = AudioPreprocessor(PreprocessConfig(), sample_rate=32_000)
    audio = pre.load(wav_file)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    # ~12 s at 32 kHz.
    assert abs(audio.size - 32_000 * 12) < 32_000


def test_iter_windows_count(wav_file: Path) -> None:
    pre = AudioPreprocessor(PreprocessConfig(), sample_rate=32_000)
    windows = list(pre.iter_windows(wav_file, window_s=5.0, hop_s=5.0))
    # 12 s file, 5 s windows at 5 s hop -> starts at 0, 5 (last_start=7 -> 0,5).
    assert [round(w.start_s) for w in windows] == [0, 5]
    assert all(w.waveform.size == 32_000 * 5 for w in windows)


def test_preprocess_config_normalize_resampler() -> None:
    import pytest
    from pydantic import ValidationError

    cfg = PreprocessConfig()
    assert cfg.normalize == "hoplite" and cfg.resampler == "polyphase"  # canonical defaults
    assert PreprocessConfig(normalize="none", resampler="soxr_hq").normalize == "none"
    with pytest.raises(ValidationError):
        PreprocessConfig(normalize="bogus")


def test_resampler_choice_changes_samples(tmp_path: Path) -> None:
    # A 24 kHz file so resampling to 32 kHz actually runs.
    sr = 24_000
    rng = np.random.default_rng(0)
    wav = tmp_path / "PIC02_20250530_040000.wav"
    sf.write(wav, rng.normal(0, 0.1, sr * 6).astype(np.float32), sr)

    def first_window(resampler: str) -> np.ndarray:
        pre = AudioPreprocessor(PreprocessConfig(resampler=resampler), sample_rate=32_000)
        return next(iter(pre.iter_windows(wav, window_s=5.0, hop_s=5.0))).waveform

    poly, soxr = first_window("polyphase"), first_window("soxr_hq")
    # Both yield correct-length 32 kHz float32 windows...
    assert poly.size == soxr.size == 32_000 * 5
    assert poly.dtype == soxr.dtype == np.float32
    # ...but the filter choice genuinely changes the samples.
    assert not np.allclose(poly, soxr)


class _FakeHopliteModel:
    """Minimal stand-in exposing the ``target_peak`` attribute InferenceEngine sets."""

    sample_rate = 32_000
    target_peak: float | None = 0.25


def _perch_model_around(raw: object) -> PerchModel:
    return PerchModel(
        model=raw, name="fake", sample_rate=32_000, window_size_s=5.0, hop_size_s=5.0,
        embedding_dim=4, logits_key="label", class_names=["a"],
    )


def test_inference_engine_applies_normalize_choice() -> None:
    # normalize="none" disables the model's peak-normalization.
    pm = _perch_model_around(_FakeHopliteModel())
    pre = AudioPreprocessor(PreprocessConfig(normalize="none"), sample_rate=32_000)
    InferenceEngine(pm, pre, window_s=5.0, hop_s=5.0, batch_size=8)
    assert pm.model.target_peak is None

    # normalize="hoplite" leaves the model's native target_peak untouched.
    pm2 = _perch_model_around(_FakeHopliteModel())
    pre2 = AudioPreprocessor(PreprocessConfig(normalize="hoplite"), sample_rate=32_000)
    InferenceEngine(pm2, pre2, window_s=5.0, hop_s=5.0, batch_size=8)
    assert pm2.model.target_peak == 0.25


def test_set_target_peak_safe_without_attribute(perch_placeholder: PerchModel) -> None:
    # The placeholder model has no target_peak; setting it is a harmless no-op.
    perch_placeholder.set_target_peak(None)
    assert not hasattr(perch_placeholder.model, "target_peak")


def test_sigmoid_bounds() -> None:
    out = sigmoid(np.array([-1000.0, 0.0, 1000.0], dtype=np.float32))
    assert np.isclose(out[0], 0.0) and np.isclose(out[1], 0.5) and np.isclose(out[2], 1.0)


def test_softmax_normalises_and_desaturates() -> None:
    # Large competing logits: softmax normalises and keeps the winner < 1.0,
    # where sigmoid would saturate all of them to ~1.0.
    out = softmax(np.array([10.0, 8.0, 9.0], dtype=np.float32))
    assert np.isclose(out.sum(), 1.0)
    assert int(np.argmax(out)) == 0 and out[0] < 0.9


def test_classifier_default_activation_is_softmax() -> None:
    taxonomy = TaxonomyMap(["sp_a", "sp_b", "sp_c"])
    runner = ClassifierRunner(taxonomy, top_k=3)  # default activation
    results = [
        WindowResult(0.0, 5.0, np.zeros(4, np.float32), np.array([0.0, 2.0, -1.0], np.float32))
    ]
    ranked = runner.predict_windows(results)[0].ranked
    assert [r.class_index for r in ranked] == [1, 0, 2]  # ranked by confidence
    assert np.isclose(sum(r.confidence for r in ranked), 1.0)  # softmax normalises


def test_classifier_topk_and_threshold() -> None:
    # Sigmoid activation; logits chosen so confidences are ~ [0.5, 0.88, 0.27].
    taxonomy = TaxonomyMap(["sp_a", "sp_b", "sp_c"])
    runner = ClassifierRunner(taxonomy, top_k=2, activation="sigmoid")
    results = [
        WindowResult(
            start_s=0.0,
            end_s=5.0,
            embedding=np.zeros(4, np.float32),
            logits=np.array([0.0, 2.0, -1.0], np.float32),
        )
    ]
    cache = runner.predict_windows(results)
    assert len(cache) == 1
    ranked = cache[0].ranked
    assert [r.class_index for r in ranked] == [1, 0]  # top-2 by confidence

    meta = parse_filename(Path("/x/PIC02_20250530_040000.wav"), FilenameConfig())
    high = runner.detections_at_threshold(cache, recording=meta, threshold=0.8, window_s=5, hop_s=5)
    assert [d.label for d in high] == ["sp_b"]  # only sp_b >= 0.8
    low = runner.detections_at_threshold(cache, recording=meta, threshold=0.4, window_s=5, hop_s=5)
    assert [d.label for d in low] == ["sp_b", "sp_a"]  # cache reused, both pass


def test_writers_roundtrip(tmp_path: Path) -> None:
    taxonomy = TaxonomyMap(["Ardea cinerea", "Pelophylax perezi"])
    runner = ClassifierRunner(taxonomy, top_k=1)
    results = [
        WindowResult(0.0, 5.0, np.zeros(2, np.float32), np.array([5.0, -5.0], np.float32)),
        WindowResult(5.0, 10.0, np.zeros(2, np.float32), np.array([-5.0, 5.0], np.float32)),
    ]
    cache = runner.predict_windows(results)
    meta = parse_filename(Path("/x/PIC02_20250530_040000.wav"), FilenameConfig())
    dets = runner.detections_at_threshold(cache, recording=meta, threshold=0.6, window_s=5, hop_s=5)
    assert len(dets) == 2

    raven_path = write_selection_table(dets, tmp_path / "out.txt")
    df = read_selection_table(raven_path)
    assert list(df.columns) == RAVEN_COLUMNS
    assert df["Common Name"].tolist() == ["Ardea cinerea", "Pelophylax perezi"]

    csv_path = write_csv(dets, tmp_path / "out.csv")
    import pandas as pd

    csv = pd.read_csv(csv_path)
    assert list(csv.columns) == CSV_COLUMNS
    assert csv["date"].tolist() == ["2025-05-30", "2025-05-30"]


def _detection_in_file(wav: Path, *, start_s: float, end_s: float) -> Detection:
    meta = parse_filename(wav, FilenameConfig())
    return Detection(
        recording=meta, start_s=start_s, end_s=end_s, window_s=5.0, hop_s=5.0, rank=1,
        label="Ardea cinerea", common_name="Ardea cinerea", species_code="Ardea cinerea",
        confidence=0.9, threshold=0.1, expected_label=None,
    )


def _clip_frames(out_dir: Path) -> int:
    clips = list((out_dir / "Ardea_cinerea").glob("*.wav"))
    assert len(clips) == 1
    return int(sf.info(str(clips[0])).frames)


def test_extract_segments_context_padding(tmp_path: Path) -> None:
    sr = 32_000
    wav = tmp_path / "PIC02_20250530_040000.wav"
    sf.write(wav, np.zeros(sr * 20, dtype=np.float32), sr)  # 20 s file
    det = _detection_in_file(wav, start_s=5.0, end_s=10.0)  # a 5 s detection window

    # No context: clip is exactly clip_duration_s.
    plain = extract_segments(
        [det], sample_rate=sr,
        config=SegmentConfig(enabled=True, clip_duration_s=5.0, seed=0),
        output_dir=tmp_path / "plain",
    )
    assert plain.clips_written == 1
    assert _clip_frames(tmp_path / "plain") == 5 * sr

    # 1 s of context per side: 5 s centre -> 7 s total.
    padded = extract_segments(
        [det], sample_rate=sr,
        config=SegmentConfig(enabled=True, clip_duration_s=5.0, context_s=1.0, seed=0),
        output_dir=tmp_path / "padded",
    )
    assert padded.clips_written == 1
    assert _clip_frames(tmp_path / "padded") == 7 * sr
