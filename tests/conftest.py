"""Shared pytest fixtures.

The heavy Perch V2 model requires a Kaggle download, so unit tests use Hoplite's
``PlaceholderModel`` wrapped in :class:`perchlab.models.PerchModel`. Tests that
need the real model are marked ``slow`` and skipped by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from perchlab.config import PERCH_SAMPLE_RATE_HZ
from perchlab.models import PerchModel


def make_placeholder_model(
    *,
    sample_rate: int = PERCH_SAMPLE_RATE_HZ,
    window_size_s: float = 5.0,
    embedding_dim: int = 1536,
) -> PerchModel:
    """Build a :class:`PerchModel` backed by Hoplite's PlaceholderModel."""
    from perch_hoplite.zoo.placeholder_model import PlaceholderModel

    raw = PlaceholderModel(
        sample_rate=sample_rate,
        window_size_s=window_size_s,
        hop_size_s=window_size_s,
        embedding_size=embedding_dim,
        do_frame_audio=False,
    )
    return PerchModel._from_loaded(raw, name="placeholder", embedding_dim=embedding_dim)


@pytest.fixture
def perch_placeholder() -> PerchModel:
    """A fast, offline stand-in for Perch V2."""
    return make_placeholder_model()


@pytest.fixture
def silence_5s() -> np.ndarray:
    """Five seconds of 32 kHz float32 silence."""
    return np.zeros(PERCH_SAMPLE_RATE_HZ * 5, dtype=np.float32)
