"""Tests for the PerchModel wrapper (placeholder-backed + optional real model)."""

from __future__ import annotations

import numpy as np
import pytest

from perchlab.config import ModelConfig
from perchlab.models import PerchModel, _resolve_preset_name


def test_placeholder_model_shapes(perch_placeholder: PerchModel, silence_5s: np.ndarray) -> None:
    model = perch_placeholder
    assert model.sample_rate == 32_000
    assert model.embedding_dim == 1536
    assert model.num_classes > 0
    assert model.logits_key == "label"

    out = model.embed(silence_5s)
    emb = model.window_embeddings(out)
    logits = model.window_logits(out)

    assert emb.ndim == 2 and emb.shape[1] == 1536
    assert logits.ndim == 2 and logits.shape[1] == model.num_classes
    assert emb.shape[0] == logits.shape[0]  # same number of windows


def test_class_names_align_with_logits(
    perch_placeholder: PerchModel, silence_5s: np.ndarray
) -> None:
    out = perch_placeholder.embed(silence_5s)
    logits = perch_placeholder.window_logits(out)
    assert logits.shape[1] == len(perch_placeholder.class_names)


def test_resolve_preset_name() -> None:
    assert _resolve_preset_name(ModelConfig(backend="auto")) == "perch_v2"
    assert _resolve_preset_name(ModelConfig(backend="onnx")) == "perch_v2_onnx"
    assert _resolve_preset_name(ModelConfig(backend="cpu")) == "perch_v2_cpu"
    assert _resolve_preset_name(ModelConfig(name="SurfPerch")) == "surfperch"


@pytest.mark.slow
def test_real_perch_v2_loads() -> None:
    """Only runs when explicitly selected (needs Kaggle download)."""
    model = PerchModel.load(ModelConfig(backend="onnx"))
    assert model.num_classes > 10_000
    out = model.embed(np.zeros(model.sample_rate * 5, dtype=np.float32))
    assert model.window_logits(out).shape[1] == model.num_classes
