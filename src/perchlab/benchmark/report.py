"""Write machine- and human-readable benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..logging import get_logger
from .metrics import CurveMetrics, ThresholdMetrics, metrics_summary

_log = get_logger("benchmark.report")


def write_machine_readable(
    output_dir: Path,
    *,
    primary: ThresholdMetrics,
    curves: CurveMetrics,
    sweep_table: pd.DataFrame,
) -> list[Path]:
    """Write JSON/CSV artefacts and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    summary_path = output_dir / "metrics.json"
    summary_json = json.dumps(metrics_summary(primary, curves), indent=2)
    summary_path.write_text(summary_json, encoding="utf-8")
    written.append(summary_path)

    per_class_rows = []
    for label, vals in primary.per_class.items():
        per_class_rows.append(
            {
                "label": label,
                **vals,
                "roc_auc": curves.per_class_roc_auc.get(label),
                "pr_auc": curves.per_class_pr_auc.get(label),
            }
        )
    per_class_path = output_dir / "metrics.csv"
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)
    written.append(per_class_path)

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(primary.report_text, encoding="utf-8")
    written.append(report_path)

    cm_path = output_dir / "confusion_matrix.csv"
    labels = primary.confusion_labels
    pd.DataFrame(primary.confusion, index=labels, columns=labels).to_csv(cm_path)
    written.append(cm_path)

    sweep_path = output_dir / "sweep.csv"
    sweep_table.to_csv(sweep_path, index=False)
    written.append(sweep_path)

    return written


def write_report_markdown(
    output_dir: Path,
    *,
    primary: ThresholdMetrics,
    curves: CurveMetrics,
    plot_paths: dict[str, Path],
    n_samples: int,
    aggregate: str,
) -> Path:
    """Write a human-readable Markdown report embedding the plots."""
    lines = [
        "# PerchLab Benchmark Report",
        "",
        f"- Samples evaluated: **{n_samples}** ({aggregate}-level)",
        f"- Primary threshold: **{primary.threshold:.2f}**",
        f"- Accuracy: **{primary.accuracy:.3f}**",
        f"- Macro F1: **{primary.f1_macro:.3f}**  |  Micro F1: **{primary.f1_micro:.3f}**",
        f"- Macro ROC-AUC: **{curves.roc_auc_macro:.3f}**  |  "
        f"Macro PR-AUC: **{curves.pr_auc_macro:.3f}**",
        "",
        "## Classification report",
        "",
        "```",
        primary.report_text,
        "```",
        "",
        "## Figures",
        "",
    ]
    for title, path in plot_paths.items():
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({path.name})")
        lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
