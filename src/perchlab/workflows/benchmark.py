"""Workflow 3 - Benchmark.

Evaluate Perch V2 on a labelled dataset (parent folder = species label, top-1 vs
expected). Produces accuracy/precision/recall/F1, a confusion matrix, sklearn's
classification report, per-class ROC/PR curves + AUCs, metric-vs-threshold
plots, and both machine- and human-readable reports.
"""

from __future__ import annotations

from pathlib import Path

from ..benchmark import dataset, evaluate, metrics, plots, report, sweep
from ..config import AppConfig
from ..errors import WorkflowError
from ..inference import InferenceEngine
from ..logging import get_logger
from ..preprocess import AudioPreprocessor
from ..util import default_output_dir, set_global_seed, write_manifest
from .base import RunSummary, Workflow

_log = get_logger("workflow.benchmark")

_INFO_MESSAGE = (
    "Benchmark requires a labelled dataset: a folder named as the species label, "
    "or a folder containing one subfolder per species."
)


class BenchmarkWorkflow(Workflow):
    """Evaluate Perch on a folder-labelled dataset."""

    name = "Benchmark"
    command = "benchmark"
    description = "Evaluate Perch on a labelled dataset (top-1 classification)."

    def configure_interactive(self, config: AppConfig) -> AppConfig:
        """Prompt for benchmark parameters."""
        from .. import prompts  # noqa: PLC0415

        _log.info(_INFO_MESSAGE)
        cfg = config.benchmark
        cfg.input_dir = prompts.ask_path("Labelled dataset folder:", must_exist=True)
        default_out = str(default_output_dir("benchmark"))
        cfg.output_dir = prompts.ask_path("Output folder:", default=default_out, must_exist=False)
        cfg.window_s = prompts.ask_float("Window size (s):", default=cfg.window_s)
        cfg.hop_s = prompts.ask_float("Hop size (s):", default=cfg.hop_s)
        if prompts.ask_bool("Sweep multiple thresholds?", default=False):
            cfg.sweep.enabled = True
            cfg.sweep.start = prompts.ask_float("  Start threshold:", default=0.0)
            cfg.sweep.end = prompts.ask_float("  End threshold:", default=1.0)
            cfg.sweep.step = prompts.ask_float("  Step:", default=0.1)
        else:
            cfg.threshold = prompts.ask_float("Confidence threshold:", default=cfg.threshold)
        return config

    def run(self, config: AppConfig) -> RunSummary:
        """Execute the classification benchmark."""
        set_global_seed(config.seed)
        cfg = config.benchmark
        if cfg.mode != "classification":
            raise WorkflowError(
                f"Benchmark mode '{cfg.mode}' is not implemented yet (classification only)."
            )
        if cfg.input_dir is None:
            raise WorkflowError("No input folder configured.")
        _log.info(_INFO_MESSAGE)
        output_dir = Path(cfg.output_dir or default_output_dir("benchmark"))
        output_dir.mkdir(parents=True, exist_ok=True)

        thresholds = cfg.sweep.values() if cfg.sweep.enabled else [cfg.threshold]
        # Report the full metric set at the threshold closest to the configured one.
        primary_threshold = min(thresholds, key=lambda t: abs(t - cfg.threshold))
        self.log_parameters(
            {
                "input": cfg.input_dir,
                "output": output_dir,
                "window_s": cfg.window_s,
                "hop_s": cfg.hop_s,
                "top_k": 1,
                "aggregate": cfg.aggregate,
                "thresholds": thresholds,
            }
        )
        write_manifest(output_dir, workflow=self.name, config=config.model_dump(mode="json"))

        files = dataset.load_labelled_dataset(cfg.input_dir, config.filename)
        model = self.load_model(config.model)
        preprocessor = AudioPreprocessor(config.preprocess, model.sample_rate)
        engine = InferenceEngine(
            model,
            preprocessor,
            window_s=cfg.window_s,
            hop_s=cfg.hop_s,
            batch_size=config.model.batch_size,
        )

        _log.info("Running inference over %d files ...", len(files))
        data = evaluate.evaluate_dataset(
            files, model, engine, aggregate=cfg.aggregate, activation=config.model.activation
        )
        if not data.y_true:
            raise WorkflowError("No windows were evaluated; check the dataset and labels.")

        sweep_table, per_threshold = sweep.run_sweep(data, thresholds)
        primary = per_threshold[primary_threshold]
        curves = metrics.compute_curve_metrics(data)

        written = report.write_machine_readable(
            output_dir, primary=primary, curves=curves, sweep_table=sweep_table
        )
        plot_paths = {
            "Confusion matrix": plots.plot_confusion_matrix(
                primary, output_dir / "confusion_matrix.png"
            ),
            "ROC curves": plots.plot_roc_curves(curves, output_dir / "roc_curves.png"),
            "Precision-Recall curves": plots.plot_pr_curves(curves, output_dir / "pr_curves.png"),
            "Metrics vs threshold": plots.plot_metric_vs_threshold(
                sweep_table, output_dir / "metrics_vs_threshold.png"
            ),
        }
        report_md = report.write_report_markdown(
            output_dir,
            primary=primary,
            curves=curves,
            plot_paths=plot_paths,
            n_samples=len(data.y_true),
            aggregate=data.aggregate,
        )

        summary = RunSummary(workflow=self.name)
        summary.processed = len(files)
        summary.detections = len(data.y_true)
        for path in [*written, *plot_paths.values(), report_md]:
            summary.add_output(path)
        _log.info(
            "Accuracy=%.3f  macroF1=%.3f  ROC-AUC=%.3f (threshold=%.2f)",
            primary.accuracy,
            primary.f1_macro,
            curves.roc_auc_macro,
            primary.threshold,
        )
        summary.log_final()
        return summary
