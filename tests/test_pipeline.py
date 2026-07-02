"""Tests for audio parsing, preprocessing, classification, and IO."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from perchlab.audio import discover_audio, parse_filename
from perchlab.classify import ClassifierRunner, sigmoid
from perchlab.config import FilenameConfig, PreprocessConfig
from perchlab.detections import CSV_COLUMNS
from perchlab.inference import WindowResult
from perchlab.io.raven import RAVEN_COLUMNS, read_selection_table, write_selection_table
from perchlab.io.tables import write_csv
from perchlab.preprocess import AudioPreprocessor
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


def test_sigmoid_bounds() -> None:
    out = sigmoid(np.array([-1000.0, 0.0, 1000.0], dtype=np.float32))
    assert np.isclose(out[0], 0.0) and np.isclose(out[1], 0.5) and np.isclose(out[2], 1.0)


def test_classifier_topk_and_threshold() -> None:
    # Three classes; logits chosen so confidences are ~ [0.5, 0.88, 0.27].
    taxonomy = TaxonomyMap(["sp_a", "sp_b", "sp_c"])
    runner = ClassifierRunner(taxonomy, top_k=2)
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
