"""Tests for optimal-confidence-threshold statistics and dataset loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perchlab.errors import ThresholdError
from perchlab.threshold.dataset import load_validated_dataset
from perchlab.threshold.stats import (
    confidence_from_logit,
    fit_species_threshold,
    logit_score,
    precision_bins,
)


def test_logit_transform_roundtrips() -> None:
    conf = np.array([0.05, 0.2, 0.5, 0.9, 0.999])
    back = np.array([confidence_from_logit(v) for v in logit_score(conf)])
    assert np.allclose(back, conf, atol=1e-6)


def test_logit_score_is_finite_at_boundaries() -> None:
    # ln(c/(1-c)) diverges at both 0 and 1; clipping keeps it finite and
    # antisymmetric, and is exactly 0 at c=0.5.
    assert np.isfinite(logit_score(np.array([0.0]))[0])
    assert np.isfinite(logit_score(np.array([1.0]))[0])
    assert logit_score(np.array([0.5]))[0] == pytest.approx(0.0, abs=1e-9)


def test_precision_bins_match_reference_shape() -> None:
    # Two low-confidence wrong, rest correct across the three default categories.
    conf = np.array([0.15, 0.25, 0.35, 0.45, 0.6, 0.8])
    correct = np.array([False, True, True, True, True, True])
    bins = precision_bins(conf, correct, [0.1, 0.3, 0.5])
    assert [b.category for b in bins] == ["[0.10, 0.30)", "[0.30, 0.50)", "[0.50, 1.00]"]
    assert [b.detections for b in bins] == [2, 2, 2]
    assert [b.verified for b in bins] == [1, 2, 2]
    assert np.isclose(bins[0].precision, 0.5)


def test_fit_recovers_known_threshold() -> None:
    """A large sample from a known logistic model recovers its 95% threshold."""
    rng = np.random.default_rng(0)
    b0, b1 = 1.0, 1.5
    n = 6000
    conf = rng.uniform(0.001, 0.999, n)
    p = 1.0 / (1.0 + np.exp(-(b0 + b1 * logit_score(conf))))
    correct = rng.uniform(size=n) < p

    result = fit_species_threshold("Test sp", conf, correct, target_probability=0.95)

    analytic = confidence_from_logit((np.log(0.95 / 0.05) - b0) / b1)
    assert result.fitted
    assert abs(result.threshold - analytic) < 0.05
    # Detections kept by the threshold are correct with ~>= the target precision.
    kept = conf >= result.threshold
    assert correct[kept].mean() >= 0.93


def test_higher_target_probability_gives_higher_threshold() -> None:
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.001, 0.999, 4000)
    p = 1.0 / (1.0 + np.exp(-(0.5 + 1.2 * logit_score(conf))))
    correct = rng.uniform(size=conf.size) < p
    t90 = fit_species_threshold("s", conf, correct, target_probability=0.90).threshold
    t99 = fit_species_threshold("s", conf, correct, target_probability=0.99).threshold
    assert t99 > t90


def test_degenerate_all_correct_is_not_fitted() -> None:
    conf = np.linspace(0.1, 0.9, 20)
    result = fit_species_threshold("s", conf, np.ones(20, dtype=bool))
    assert not result.fitted and np.isnan(result.threshold)
    assert "all detections are correct" in result.note


def test_degenerate_too_few_samples() -> None:
    result = fit_species_threshold("s", np.array([0.2, 0.8]), np.array([False, True]))
    assert not result.fitted and "need >=" in result.note


def test_anticorrelated_confidence_is_not_fitted() -> None:
    # Correctness decreases with confidence -> slope <= 0 -> no usable threshold.
    conf = np.linspace(0.05, 0.95, 200)
    correct = conf < 0.5
    result = fit_species_threshold("s", conf, correct)
    assert not result.fitted and "slope <= 0" in result.note


def _wav(path: Path) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(32_000 * 5, dtype=np.float32), 32_000)


def test_load_single_species_layout(tmp_path: Path) -> None:
    _wav(tmp_path / "correct" / "a.wav")
    _wav(tmp_path / "incorrect" / "b.wav")
    files = load_validated_dataset(tmp_path, species="Periparus ater")
    assert {f.correct for f in files} == {True, False}
    assert all(f.species == "Periparus ater" for f in files)


def test_load_multi_species_layout(tmp_path: Path) -> None:
    _wav(tmp_path / "Periparus ater" / "present" / "a.wav")
    _wav(tmp_path / "Periparus ater" / "absent" / "b.wav")
    _wav(tmp_path / "Certhia brachydactyla" / "true" / "c.wav")
    _wav(tmp_path / "Certhia brachydactyla" / "false" / "d.wav")
    files = load_validated_dataset(tmp_path, species=None)
    assert {f.species for f in files} == {"Periparus ater", "Certhia brachydactyla"}


def test_single_species_layout_requires_species(tmp_path: Path) -> None:
    _wav(tmp_path / "correct" / "a.wav")
    with pytest.raises(ThresholdError, match="pass --species"):
        load_validated_dataset(tmp_path, species=None)
