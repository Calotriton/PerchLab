"""Workflow 4 - Optimal Confidence Threshold Detection.

Estimate, per species, the Perch confidence threshold that corresponds to a
target probability (default 95%) of a detection being correct. The input is a set
of human-validated detections sorted into correct/incorrect folders; the workflow
runs Perch to recover each clip's confidence, fits a logistic regression of
correctness on the logit-transformed confidence, and inverts it to the threshold.
Outputs a precision table and a probability-of-correct plot per species.
"""

from __future__ import annotations

from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from ..config import AppConfig
from ..errors import ThresholdError
from ..inference import InferenceEngine
from ..logging import console, get_logger
from ..preprocess import AudioPreprocessor
from ..threshold import collect, dataset, plots, report, stats
from ..util import default_output_dir, set_global_seed, write_manifest
from .base import RunSummary, Workflow

_log = get_logger("workflow.threshold")

_INFO_MESSAGE = (
    "Optimal threshold needs human-validated detections: correct/ and incorrect/ "
    "subfolders (per species when a dataset spans several)."
)


class OptimalThresholdWorkflow(Workflow):
    """Estimate species-specific confidence thresholds from validated detections."""

    name = "Optimal Confidence Threshold Detection"
    command = "threshold"
    description = "Estimate the confidence threshold for a target precision, per species."

    def configure_interactive(self, config: AppConfig) -> AppConfig:
        """Prompt for optimal-threshold parameters."""
        from .. import prompts  # noqa: PLC0415

        _log.info(_INFO_MESSAGE)
        cfg = config.optimal_threshold
        cfg.input_dir = prompts.ask_path("Validated dataset folder:", must_exist=True)
        default_out = str(default_output_dir("perchlab_threshold"))
        cfg.output_dir = prompts.ask_path("Output folder:", default=default_out, must_exist=False)
        species = prompts.ask_text(
            "Target species (blank = one per subfolder):", default=cfg.species or ""
        )
        cfg.species = species or None
        cfg.target_probability = prompts.ask_float(
            "Target probability of correct identification:", default=cfg.target_probability
        )
        return config

    def run(self, config: AppConfig) -> RunSummary:
        """Estimate thresholds and write the table, plot, and report."""
        set_global_seed(config.seed)
        cfg = config.optimal_threshold
        if cfg.input_dir is None:
            raise ThresholdError("No input folder configured.")
        _log.info(_INFO_MESSAGE)
        output_dir = Path(cfg.output_dir or default_output_dir("perchlab_threshold"))
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log_parameters({
            "input": cfg.input_dir,
            "output": output_dir,
            "species": cfg.species or "(per subfolder)",
            "target_probability": cfg.target_probability,
            "window_s": cfg.window_s,
            "hop_s": cfg.hop_s,
        })
        write_manifest(output_dir, workflow=self.name, config=config.model_dump(mode="json"))

        files = dataset.load_validated_dataset(cfg.input_dir, species=cfg.species)
        model = self.load_model(config.model)
        preprocessor = AudioPreprocessor(config.preprocess, model.sample_rate)
        engine = InferenceEngine(
            model, preprocessor,
            window_s=cfg.window_s, hop_s=cfg.hop_s, batch_size=config.model.batch_size,
        )

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console,
        ) as progress:
            task = progress.add_task("Scoring clips", total=len(files))
            per_species = collect.collect_scores(
                files, model, engine,
                activation=config.model.activation,
                progress_advance=lambda: progress.advance(task),
            )

        results: list[stats.SpeciesThreshold] = []
        for species in sorted(per_species):
            conf, correct = per_species[species].as_arrays()
            result = stats.fit_species_threshold(
                species, conf, correct,
                target_probability=cfg.target_probability, bin_edges=cfg.bin_edges,
            )
            results.append(result)
            if result.fitted:
                _log.info("%s: threshold=%.3f (n=%d, %d correct)",
                          species, result.threshold, result.n, result.n_correct)
            else:
                _log.warning("%s: no threshold (%s)", species, result.note)

        if not results:
            raise ThresholdError("No species could be scored; check the dataset layout.")

        plot_path = plots.plot_probability_curves(results, output_dir / "probability_curves.png")
        written = report.write_outputs(output_dir, results, plot_name=plot_path.name)

        summary = RunSummary(workflow=self.name)
        summary.processed = len(files)
        summary.detections = sum(r.n for r in results)
        for path in [*written, plot_path]:
            summary.add_output(path)
        summary.log_final()
        return summary
