"""Workflow 2 - Embedding Generation.

Generate Perch V2 embeddings for a corpus and store them in a Hoplite SQLite DB
(optionally exported to Parquet/NPZ). When ``labeled`` is set, the per-species
input folder names become embedding labels.
"""

from __future__ import annotations

from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from ..audio import discover_audio, parse_filename
from ..config import AppConfig
from ..embedding import EmbeddingRunner
from ..errors import AudioError, WorkflowError
from ..inference import InferenceEngine
from ..logging import console, get_logger
from ..preprocess import AudioPreprocessor
from ..util import default_output_dir, set_global_seed, write_manifest
from .base import RunSummary, Workflow

_log = get_logger("workflow.embed")


class EmbeddingWorkflow(Workflow):
    """Generate and persist embeddings for reuse."""

    name = "Embedding Generation"
    command = "embed"
    description = "Generate Perch embeddings and store them in a Hoplite DB."

    def configure_interactive(self, config: AppConfig) -> AppConfig:
        """Prompt for embedding parameters."""
        from .. import prompts  # noqa: PLC0415

        cfg = config.embed
        cfg.labeled = prompts.ask_bool(
            "Labeled dataset (one folder per species)?", default=False
        )
        cfg.input_dir = prompts.ask_path("Input folder:", must_exist=True)
        default_out = str(default_output_dir("embeddings"))
        cfg.output_dir = prompts.ask_path("Output folder:", default=default_out, must_exist=False)
        cfg.window_s = prompts.ask_float("Window size (s):", default=cfg.window_s)
        cfg.hop_s = prompts.ask_float("Hop size (s):", default=cfg.hop_s)
        export = prompts.select(
            "Portable export in addition to the Hoplite DB?",
            choices=["none", "parquet", "npz"],
            default="none",
        )
        cfg.export = export  # type: ignore[assignment]
        return config

    def run(self, config: AppConfig) -> RunSummary:
        """Embed the corpus into a Hoplite DB."""
        set_global_seed(config.seed)
        cfg = config.embed
        if cfg.input_dir is None:
            raise WorkflowError("No input folder configured.")
        output_dir = Path(cfg.output_dir or default_output_dir("embeddings"))
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log_parameters(
            {
                "input": cfg.input_dir,
                "output": output_dir,
                "window_s": cfg.window_s,
                "hop_s": cfg.hop_s,
                "labeled": cfg.labeled,
                "export": cfg.export,
            }
        )
        write_manifest(output_dir, workflow=self.name, config=config.model_dump(mode="json"))

        files = discover_audio(cfg.input_dir)
        if not files:
            raise WorkflowError(f"No audio files found under {cfg.input_dir}")

        model = self.load_model(config.model)
        if model.embedding_dim != cfg.embedding_dim:
            _log.warning(
                "Configured embedding_dim=%d but model produces %d; using the model's.",
                cfg.embedding_dim,
                model.embedding_dim,
            )
        preprocessor = AudioPreprocessor(config.preprocess, model.sample_rate)
        engine = InferenceEngine(
            model,
            preprocessor,
            window_s=cfg.window_s,
            hop_s=cfg.hop_s,
            batch_size=config.model.batch_size,
        )
        runner = EmbeddingRunner(
            db_dir=output_dir / "hoplite_db",
            embedding_dim=model.embedding_dim,
            labeled=cfg.labeled,
        )

        summary = RunSummary(workflow=self.name)
        want_export = cfg.export != "none"
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding", total=len(files))
            for path in files:
                meta = parse_filename(path, config.filename, input_root=cfg.input_dir)
                try:
                    windows = list(engine.run_file(path))
                except AudioError as exc:
                    _log.warning("Skipping %s: %s", path.name, exc)
                    summary.failed += 1
                    progress.advance(task)
                    continue
                inserted = runner.add_file(meta, windows, collect_export=want_export)
                summary.processed += 1
                summary.detections += inserted
                progress.advance(task)

        runner.commit()
        summary.add_output(output_dir / "hoplite_db")
        if want_export:
            ext = "parquet" if cfg.export == "parquet" else "npz"
            export_path = runner.export(output_dir / f"embeddings.{ext}", cfg.export)
            summary.add_output(export_path)

        summary.log_final()
        return summary
