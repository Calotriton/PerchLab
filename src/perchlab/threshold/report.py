"""Machine- and human-readable outputs for the optimal-threshold workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..logging import get_logger
from .stats import SpeciesThreshold

_log = get_logger("threshold.report")


def _precision_rows(result: SpeciesThreshold) -> list[dict[str, object]]:
    """Table-1-style rows (per confidence category) for one species, plus TOTAL."""
    rows: list[dict[str, object]] = []
    for b in result.bins:
        rows.append({
            "species": result.species,
            "confidence_category": b.category,
            "detections": b.detections,
            "verified": b.verified,
            "precision_pct": round(100 * b.precision, 1) if b.detections else None,
        })
    tabulated = sum(b.detections for b in result.bins)
    verified = sum(b.verified for b in result.bins)
    rows.append({
        "species": result.species,
        "confidence_category": "TOTAL",
        "detections": tabulated,
        "verified": verified,
        "precision_pct": round(100 * verified / tabulated, 1) if tabulated else None,
    })
    return rows


def write_outputs(
    output_dir: Path, results: list[SpeciesThreshold], *, plot_name: str
) -> list[Path]:
    """Write thresholds.csv/json, precision_table.csv, and report.md.

    Returns the paths written (excluding the plot, which the caller renders).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    thr_rows = [{
        "species": r.species,
        "threshold": round(r.threshold, 4) if r.fitted else None,
        "target_probability": r.target_probability,
        "n": r.n,
        "n_correct": r.n_correct,
        "n_incorrect": r.n_incorrect,
        "total_precision_pct": round(100 * r.total_precision, 1) if r.n else None,
        "intercept": round(r.intercept, 4) if r.fitted else None,
        "slope": round(r.slope, 4) if r.fitted else None,
        "fitted": r.fitted,
        "note": r.note,
    } for r in results]
    thr_csv = output_dir / "thresholds.csv"
    pd.DataFrame(thr_rows).to_csv(thr_csv, index=False)
    written.append(thr_csv)

    thr_json = output_dir / "thresholds.json"
    thr_json.write_text(json.dumps(thr_rows, indent=2), encoding="utf-8")
    written.append(thr_json)

    prec_rows = [row for r in results for row in _precision_rows(r)]
    prec_csv = output_dir / "precision_table.csv"
    pd.DataFrame(prec_rows).to_csv(prec_csv, index=False)
    written.append(prec_csv)

    written.append(_write_markdown(output_dir, results, plot_name=plot_name))
    return written


def _write_markdown(output_dir: Path, results: list[SpeciesThreshold], *, plot_name: str) -> Path:
    """Human-readable report with the per-species threshold and precision tables."""
    target = results[0].target_probability if results else 0.95
    lines = [
        "# PerchLab - Optimal Confidence Threshold",
        "",
        f"Confidence threshold for a **{target:.0%} probability of correct "
        "identification**, estimated per species by logistic regression of "
        "human-validated detections on the logit-transformed confidence score.",
        "",
        "## Estimated thresholds",
        "",
        "| Species | Threshold | n | Correct | Incorrect | Overall precision |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        thr = f"**{r.threshold:.2f}**" if r.fitted else "—"
        prec = f"{100 * r.total_precision:.1f}%" if r.n else "—"
        lines.append(
            f"| {r.species} | {thr} | {r.n} | {r.n_correct} | {r.n_incorrect} | {prec} |"
        )
        if r.note:
            lines.append(f"|   ↳ *{r.note}* | | | | | |")

    lines += ["", "## Precision by confidence category", ""]
    for r in results:
        lines += [f"### {r.species}", "",
                  "| Confidence category | Detections | Verified | Precision |",
                  "| --- | --- | --- | --- |"]
        for b in r.bins:
            prec = f"{100 * b.precision:.1f}%" if b.detections else "—"
            lines.append(f"| {b.category} | {b.detections} | {b.verified} | {prec} |")
        tot_det = sum(b.detections for b in r.bins)
        tot_ver = sum(b.verified for b in r.bins)
        tot_prec = f"{100 * tot_ver / tot_det:.1f}%" if tot_det else "—"
        lines.append(f"| **TOTAL** | {tot_det} | {tot_ver} | {tot_prec} |")
        lines.append("")

    lines += ["## Figure", "", f"![Probability of correct detection]({plot_name})", ""]
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
