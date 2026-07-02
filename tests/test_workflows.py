"""Integration tests for the three workflows using the placeholder model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from perchlab.config import AppConfig
from perchlab.workflows.base import Workflow
from tests.conftest import make_placeholder_model


@pytest.fixture(autouse=True)
def _use_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every workflow to use the offline placeholder model."""
    monkeypatch.setattr(Workflow, "load_model", staticmethod(lambda _cfg: make_placeholder_model()))


def _make_wav(path: Path, seconds: float = 12.0, sr: int = 32_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(path.name)) % 2**32)
    sf.write(path, rng.normal(0, 0.05, int(sr * seconds)).astype(np.float32), sr)


def test_identify_workflow(tmp_path: Path) -> None:
    from perchlab.workflows.identify import SpeciesIdentificationWorkflow

    in_dir = tmp_path / "recordings"
    _make_wav(in_dir / "PIC02_20250530_040000.wav")
    out_dir = tmp_path / "out"

    config = AppConfig()
    config.identify.input_dir = in_dir
    config.identify.output_dir = out_dir
    config.identify.top_k = 3
    config.identify.threshold = 0.0  # placeholder confidences are 0.5

    summary = SpeciesIdentificationWorkflow().run(config)
    assert summary.processed == 1
    thr_dir = out_dir / "threshold_0.00"
    assert (thr_dir / "PIC02_20250530_040000.csv").exists()
    assert (thr_dir / "PIC02_20250530_040000.selection.table.txt").exists()
    assert (thr_dir / "all_detections.csv").exists()
    assert (out_dir / "manifest.json").exists()


def test_identify_multithreshold_single_inference(tmp_path: Path) -> None:
    from perchlab.workflows.identify import SpeciesIdentificationWorkflow

    in_dir = tmp_path / "recordings"
    _make_wav(in_dir / "PIC03_20250619_055602.wav")
    config = AppConfig()
    config.identify.input_dir = in_dir
    config.identify.output_dir = tmp_path / "out"
    config.identify.sweep.enabled = True
    config.identify.sweep.start = 0.0
    config.identify.sweep.end = 0.6
    config.identify.sweep.step = 0.3

    summary = SpeciesIdentificationWorkflow().run(config)
    assert summary.processed == 1
    for t in ("0.00", "0.30", "0.60"):
        assert ((tmp_path / "out" / f"threshold_{t}") / "all_detections.csv").exists()


def test_embedding_workflow(tmp_path: Path) -> None:
    from perchlab.workflows.embed import EmbeddingWorkflow

    in_dir = tmp_path / "Emberiza schoeniclus"
    _make_wav(in_dir / "PIC02_20250530_040000.wav")
    config = AppConfig()
    config.embed.input_dir = in_dir
    config.embed.output_dir = tmp_path / "emb"
    config.embed.labeled = True
    config.embed.export = "parquet"

    summary = EmbeddingWorkflow().run(config)
    assert summary.processed == 1
    assert summary.detections > 0
    assert (tmp_path / "emb" / "hoplite_db").exists()
    assert (tmp_path / "emb" / "embeddings.parquet").exists()


def test_benchmark_workflow(tmp_path: Path) -> None:
    from perchlab.workflows.benchmark import BenchmarkWorkflow

    # Two species folders; labels must be real placeholder class names.
    model = make_placeholder_model()
    label_a, label_b = model.class_names[0], model.class_names[1]
    for label in (label_a, label_b):
        for i in range(2):
            _make_wav(tmp_path / "data" / label / f"PIC02_2025053{i}_040000.wav", seconds=6.0)

    config = AppConfig()
    config.benchmark.input_dir = tmp_path / "data"
    config.benchmark.output_dir = tmp_path / "bench"
    config.benchmark.sweep.enabled = True
    config.benchmark.sweep.start = 0.0
    config.benchmark.sweep.end = 1.0
    config.benchmark.sweep.step = 0.5

    summary = BenchmarkWorkflow().run(config)
    assert summary.processed == 4
    out = tmp_path / "bench"
    for name in (
        "metrics.json",
        "metrics.csv",
        "classification_report.txt",
        "confusion_matrix.csv",
        "sweep.csv",
        "report.md",
    ):
        assert (out / name).exists(), name
    for fig in (
        "confusion_matrix.png",
        "roc_curves.png",
        "pr_curves.png",
        "metrics_vs_threshold.png",
    ):
        assert (out / fig).exists(), fig
