"""Statistics for the optimal-confidence-threshold workflow.

For one species we have a set of human-validated detections, each with the
confidence score Perch assigned and a boolean *correct* (species truly present)
label. This module:

1. back-transforms confidence to the logit scale ``L = ln(c / (1 - c))`` (the
   standard logit / inverse-sigmoid, per Wood et al. 2023);
2. fits a logistic regression ``P(correct) = sigmoid(b0 + b1 * L)`` (MLE, no
   regularisation) of correctness on that logit score;
3. inverts the fitted model to the confidence ``c*`` at which a detection has a
   target probability (default 0.95) of being correct -- the species threshold;
4. builds the binned precision table (detections / verified / precision %).

The transform + inversion were validated against Wood et al. (2023)'s method and
the raw validation dataset of Bota et al. (2023): the inversion recovers a fitted
P(correct) of exactly the target at ``c*`` on the real data.

The pure-numpy/sklearn functions here take arrays, not audio, so the estimation
is unit-testable without a model. See the README for the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..logging import get_logger

_log = get_logger("threshold.stats")

#: Confidences are clipped into ``[EPS, 1 - EPS]`` before the logit transform so
#: ``ln(c / (1 - c))`` stays finite when Perch returns a score at the boundary.
EPS = 1e-6


def logit_score(confidence: np.ndarray) -> np.ndarray:
    """Back-transform confidence to the standard logit scale ``L = ln(c / (1 - c))``.

    This is the inverse sigmoid (Wood et al. 2023): for a sigmoid-derived score it
    recovers the model's original logit, and for any score in ``[0, 1]`` it is the
    canonical logistic-calibration (Platt-scaling) link. Monotonic in ``c``, so a
    threshold on ``L`` maps one-to-one to a threshold on ``c``.
    """
    c = np.clip(np.asarray(confidence, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(c / (1.0 - c))


def confidence_from_logit(logit: float) -> float:
    """Invert :func:`logit_score`: ``c = sigmoid(L) = 1 / (1 + exp(-L))``."""
    return float(_sigmoid(np.asarray(logit, dtype=np.float64)))


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    """Logistic sigmoid with the argument clipped to avoid ``exp`` overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0)))


@dataclass
class PrecisionBin:
    """One confidence-score category in the precision table."""

    category: str
    low: float
    high: float
    detections: int
    verified: int

    @property
    def precision(self) -> float:
        """Verified / detections, or ``nan`` for an empty bin."""
        return self.verified / self.detections if self.detections else float("nan")


@dataclass
class SpeciesThreshold:
    """Fitted threshold and diagnostics for a single species."""

    species: str
    n: int
    n_correct: int
    n_incorrect: int
    target_probability: float
    #: Estimated confidence threshold (``nan`` when the fit is not usable).
    threshold: float
    intercept: float
    slope: float
    fitted: bool
    note: str = ""
    bins: list[PrecisionBin] = field(default_factory=list)
    #: Confidence grid and fitted P(correct) + 95% band, for plotting.
    curve_conf: np.ndarray = field(default_factory=lambda: np.empty(0))
    curve_prob: np.ndarray = field(default_factory=lambda: np.empty(0))
    curve_lo: np.ndarray = field(default_factory=lambda: np.empty(0))
    curve_hi: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Raw validated points (for the scatter overlay).
    points_conf: np.ndarray = field(default_factory=lambda: np.empty(0))
    points_correct: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def total_precision(self) -> float:
        """Overall verified / total detections across all validated points."""
        return self.n_correct / self.n if self.n else float("nan")


def precision_bins(
    confidence: np.ndarray, correct: np.ndarray, bin_edges: list[float]
) -> list[PrecisionBin]:
    """Bin detections by confidence and count verified (correct) ones per bin.

    Bins run ``[edge_i, edge_{i+1})`` with the final bin ``[edge_last, 1.0]``
    inclusive; detections below the first edge are not tabulated (matching the
    convention of the reference table).
    """
    conf = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct).astype(bool)
    edges = list(bin_edges) + [1.0 + EPS]
    bins: list[PrecisionBin] = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        in_bin = (conf >= low) & (conf < high)
        display_high = min(high, 1.0)
        is_last = i == len(edges) - 2
        category = f"[{low:.2f}, {display_high:.2f}{']' if is_last else ')'}"
        bins.append(
            PrecisionBin(
                category=category,
                low=low,
                high=display_high,
                detections=int(in_bin.sum()),
                verified=int((in_bin & ok).sum()),
            )
        )
    return bins


def _fit_logistic(logit: np.ndarray, correct: np.ndarray) -> tuple[float, float, np.ndarray]:
    """MLE-fit ``P(correct) = sigmoid(b0 + b1 * logit)``.

    Returns ``(intercept, slope, covariance)`` where covariance is the 2x2
    variance-covariance matrix of ``(b0, b1)`` from the observed Fisher
    information (used for the plotted confidence band).
    """
    x = logit.reshape(-1, 1)
    y = correct.astype(int)
    # C=inf => unregularised MLE, matching a standard logistic regression (the
    # supported spelling of the former penalty=None across sklearn versions).
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(x, y)
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0][0])

    # Covariance = inv(X' W X), W = diag(p(1-p)) at the fitted probabilities.
    design = np.column_stack([np.ones_like(logit), logit])
    eta = intercept + slope * logit
    p = _sigmoid(eta)
    w = np.clip(p * (1.0 - p), 1e-12, None)
    fisher = design.T @ (design * w[:, None])
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:  # pragma: no cover - near-separable data
        cov = np.linalg.pinv(fisher)
    return intercept, slope, cov


def fit_species_threshold(
    species: str,
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    target_probability: float = 0.95,
    bin_edges: list[float] | None = None,
    min_samples: int = 8,
) -> SpeciesThreshold:
    """Fit the logistic model and solve for the target-probability threshold.

    Args:
        species: Scientific name (for reporting).
        confidence: Perch confidence per validated detection, in ``[0, 1]``.
        correct: Boolean (or 0/1) correctness per detection.
        target_probability: P(correct) the threshold is solved for (default 0.95).
        bin_edges: Lower edges of the precision-table categories.
        min_samples: Minimum validated detections needed to attempt a fit.

    Returns:
        A :class:`SpeciesThreshold` with the estimate, precision bins, and the
        plot curves. ``fitted`` is ``False`` (and ``threshold`` is ``nan``) when
        the data cannot support a usable fit; ``note`` explains why.
    """
    conf = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    ok = np.asarray(correct).astype(bool)
    edges = bin_edges if bin_edges is not None else [0.1, 0.3, 0.5]
    n, n_correct = int(conf.size), int(ok.sum())
    n_incorrect = n - n_correct
    bins = precision_bins(conf, ok, edges)

    def degenerate(note: str) -> SpeciesThreshold:
        return SpeciesThreshold(
            species=species, n=n, n_correct=n_correct, n_incorrect=n_incorrect,
            target_probability=target_probability, threshold=float("nan"),
            intercept=float("nan"), slope=float("nan"), fitted=False, note=note,
            bins=bins, points_conf=conf, points_correct=ok.astype(float),
        )

    if n < min_samples:
        return degenerate(f"only {n} validated detections (need >= {min_samples})")
    if n_correct == 0 or n_incorrect == 0:
        only = "correct" if n_incorrect == 0 else "incorrect"
        return degenerate(f"all detections are {only}; cannot fit a regression")

    logit = logit_score(conf)
    intercept, slope, cov = _fit_logistic(logit, ok)

    if slope <= 0:
        result = degenerate(
            "confidence is not positively associated with correctness (slope <= 0)"
        )
        result.intercept, result.slope = intercept, slope
        return result

    # Solve sigmoid(b0 + b1 * L*) = target => L* = (logit(target) - b0) / b1,
    # then back-transform L* to a confidence threshold c* = sigmoid(L*).
    target_logit = float(np.log(target_probability / (1.0 - target_probability)))
    l_star = (target_logit - intercept) / slope
    threshold = confidence_from_logit(l_star)  # sigmoid => always in (0, 1)

    note = ""
    if threshold >= conf.max():
        note = "threshold exceeds the largest validated confidence (extrapolated)"
    elif threshold <= conf.min():
        note = "target met below the smallest validated confidence (extrapolated)"

    result = SpeciesThreshold(
        species=species, n=n, n_correct=n_correct, n_incorrect=n_incorrect,
        target_probability=target_probability, threshold=threshold,
        intercept=intercept, slope=slope, fitted=True, note=note, bins=bins,
        points_conf=conf, points_correct=ok.astype(float),
    )
    _attach_curve(result, cov)
    return result


def _attach_curve(result: SpeciesThreshold, cov: np.ndarray) -> None:
    """Populate the fitted probability curve and its 95% confidence band."""
    grid = np.linspace(EPS, 1.0 - EPS, 200)
    logit = logit_score(grid)
    design = np.column_stack([np.ones_like(logit), logit])
    eta = result.intercept + result.slope * logit
    prob = _sigmoid(eta)
    # Delta-method SE on the linear predictor, mapped through the logistic link.
    var_eta = np.einsum("ij,jk,ik->i", design, cov, design)
    se = np.sqrt(np.clip(var_eta, 0.0, None))
    lo = _sigmoid(eta - 1.96 * se)
    hi = _sigmoid(eta + 1.96 * se)
    result.curve_conf, result.curve_prob = grid, prob
    result.curve_lo, result.curve_hi = lo, hi
