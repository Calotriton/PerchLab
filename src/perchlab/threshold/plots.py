"""Figure-1-style plots for the optimal-threshold workflow.

For each species: the validated detections as a 0/1 scatter over confidence, the
fitted logistic probability-of-correct curve with its 95% band, and the
estimated threshold marked where the curve crosses the target probability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..logging import get_logger  # noqa: E402
from .stats import SpeciesThreshold  # noqa: E402

_log = get_logger("threshold.plots")


def _panel(ax: plt.Axes, result: SpeciesThreshold, rng: np.random.Generator) -> None:
    """Draw one species panel onto ``ax``."""
    conf = result.points_conf
    y = result.points_correct
    # Small vertical jitter so the 0/1 point clouds are legible.
    jitter = rng.uniform(-0.03, 0.03, size=y.shape)
    ax.scatter(conf, y + jitter, s=10, color="black", alpha=0.5, zorder=3)

    if result.fitted and result.curve_conf.size:
        ax.plot(result.curve_conf, result.curve_prob, color="tab:blue", lw=2, zorder=2)
        ax.fill_between(
            result.curve_conf, result.curve_lo, result.curve_hi,
            color="tab:blue", alpha=0.15, zorder=1,
        )
        ax.axhline(result.target_probability, color="gray", ls=":", lw=1)
        if np.isfinite(result.threshold):
            ax.axvline(result.threshold, color="tab:red", ls="--", lw=1.5, zorder=4)
            ax.annotate(
                f"threshold = {result.threshold:.2f}",
                xy=(result.threshold, 0.5), xytext=(6, 0),
                textcoords="offset points", color="tab:red", fontsize=8,
                rotation=90, va="center",
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Prob. correct detection")
    title = result.species if result.fitted else f"{result.species} (no fit)"
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


def plot_probability_curves(results: list[SpeciesThreshold], path: Path) -> Path:
    """Save a grid of per-species probability-of-correct panels."""
    n = max(1, len(results))
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False)
    rng = np.random.default_rng(0)
    flat = axes.ravel()
    for ax, result in zip(flat, results, strict=False):
        _panel(ax, result, rng)
    for ax in flat[len(results):]:
        ax.axis("off")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
