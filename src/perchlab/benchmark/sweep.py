"""Confidence-threshold sweep for the benchmark workflow."""

from __future__ import annotations

import pandas as pd

from ..logging import get_logger
from .evaluate import EvalData
from .metrics import ThresholdMetrics, compute_threshold_metrics

_log = get_logger("benchmark.sweep")


def run_sweep(
    data: EvalData, thresholds: list[float]
) -> tuple[pd.DataFrame, dict[float, ThresholdMetrics]]:
    """Compute top-1 metrics at each threshold (single inference, reused scores).

    Args:
        data: Collected evaluation arrays.
        thresholds: Thresholds to evaluate.

    Returns:
        A tuple of ``(scalar_table, per_threshold_metrics)`` where the table has
        one row per threshold (for metric-vs-threshold plots) and the dict maps
        each threshold to its full :class:`ThresholdMetrics`.
    """
    per_threshold: dict[float, ThresholdMetrics] = {}
    rows = []
    for threshold in thresholds:
        metrics = compute_threshold_metrics(data, threshold)
        per_threshold[threshold] = metrics
        rows.append(metrics.scalar_row())
    table = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    return table, per_threshold
