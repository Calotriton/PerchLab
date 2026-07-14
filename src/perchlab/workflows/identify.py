"""Workflow 1 - Species Identification.

Pipeline per file: preprocess -> one Perch forward pass -> per-window top-k ->
cache -> threshold filter -> CSV/Raven/Parquet -> optional segment extraction.
Inference runs once per file; the confidence threshold (single or a whole sweep)
is applied as a cheap post-filter over the cached predictions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from ..audio import RecordingMeta, discover_audio, is_completed, parse_filename
from ..classify import ClassifierRunner
from ..config import AppConfig, IdentifyConfig
from ..detections import CSV_COLUMNS, Detection
from ..errors import AudioError, WorkflowError
from ..inference import InferenceEngine
from ..io import raven, tables
from ..logging import console, get_logger
from ..preprocess import AudioPreprocessor
from ..segments import extract_segments
from ..taxonomy import TaxonomyMap
from ..util import default_output_dir, set_global_seed, write_manifest
from .base import RunSummary, Workflow

_log = get_logger("workflow.identify")


class SpeciesIdentificationWorkflow(Workflow):
    """Identify species in recordings using Perch V2's built-in classifier."""

    name = "Species Identification"
    command = "id"
    description = "Detect species per window and export detections."

    def configure_interactive(self, config: AppConfig) -> AppConfig:
        """Prompt for identification parameters (see :mod:`perchlab.prompts`)."""
        from .. import prompts  # local import keeps questionary out of hot paths

        cfg = config.identify
        cfg.input_dir = prompts.ask_path("Input folder (recordings):", must_exist=True)
        default_out = str(default_output_dir("perchlab_ID"))
        cfg.output_dir = prompts.ask_path(
            "Output folder:", default=default_out, must_exist=False
        )
        cfg.window_s = prompts.ask_float("Window size (s):", default=cfg.window_s)
        cfg.hop_s = prompts.ask_float("Hop size (s):", default=cfg.hop_s)
        cfg.top_k = prompts.ask_int("Top-k (predictions per window):", default=cfg.top_k)

        if prompts.ask_bool("Run multiple confidence thresholds?", default=False):
            cfg.sweep.enabled = True
            cfg.sweep.start = prompts.ask_float("  Initial threshold:", default=cfg.threshold)
            cfg.sweep.end = prompts.ask_float("  Final threshold:", default=1.0)
            cfg.sweep.step = prompts.ask_float("  Threshold step:", default=0.1)
        else:
            cfg.threshold = prompts.ask_float("Confidence threshold:", default=cfg.threshold)

        if prompts.ask_bool("Extract audio segments?", default=False):
            cfg.segments.enabled = True
            cfg.segments.bin_width = prompts.ask_float("  Confidence-bin width:", default=0.1)
            cfg.segments.max_per_bin = prompts.ask_int("  Max samples per bin:", default=20)
            cfg.segments.clip_duration_s = prompts.ask_float("  Clip duration (s):", default=5.0)
            if prompts.ask_bool("  Add context seconds around each clip?", default=False):
                cfg.segments.context_s = prompts.ask_float(
                    "    Context seconds on each side:", default=1.0
                )
            seed = prompts.ask_text("  Random seed (blank = none):", default="")
            cfg.segments.seed = int(seed) if seed else None
        return config

    def run(self, config: AppConfig) -> RunSummary:
        """Execute identification over the input folder."""
        set_global_seed(config.seed)
        cfg = config.identify
        if cfg.input_dir is None:
            raise WorkflowError("No input folder configured.")
        output_dir = Path(cfg.output_dir or default_output_dir("perchlab_ID"))
        output_dir.mkdir(parents=True, exist_ok=True)

        thresholds = cfg.sweep.values() if cfg.sweep.enabled else [cfg.threshold]
        self.log_parameters(
            {
                "input": cfg.input_dir,
                "output": output_dir,
                "window_s": cfg.window_s,
                "hop_s": cfg.hop_s,
                "top_k": cfg.top_k,
                "thresholds": thresholds,
                "formats": cfg.formats,
                "extract_segments": cfg.segments.enabled,
            }
        )
        write_manifest(output_dir, workflow=self.name, config=config.model_dump(mode="json"))

        files = discover_audio(cfg.input_dir)
        if not files:
            raise WorkflowError(f"No audio files found under {cfg.input_dir}")

        model = self.load_model(config.model)
        preprocessor = AudioPreprocessor(config.preprocess, model.sample_rate)
        engine = InferenceEngine(
            model,
            preprocessor,
            window_s=cfg.window_s,
            hop_s=cfg.hop_s,
            batch_size=config.model.batch_size,
        )
        runner = ClassifierRunner(
            TaxonomyMap(model.class_names), top_k=cfg.top_k, activation=config.model.activation
        )

        summary = RunSummary(workflow=self.name)
        thr_dirs = {t: output_dir / f"threshold_{t:.2f}" for t in thresholds}
        for path in thr_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Identifying", total=len(files))
            for path in files:
                meta = parse_filename(path, config.filename, input_root=cfg.input_dir)
                self._process_file(path, meta, runner, engine, cfg, thr_dirs, summary)
                progress.advance(task)

        # Per-threshold aggregate + summary + optional extraction (from disk, so
        # resumed/skipped files are still included).
        for threshold, thr_dir in thr_dirs.items():
            self._finalize_threshold(threshold, thr_dir, cfg, model.sample_rate, config, summary)

        summary.log_final()
        return summary

    def _process_file(
        self,
        path: Path,
        meta: RecordingMeta,
        runner: ClassifierRunner,
        engine: InferenceEngine,
        cfg: IdentifyConfig,
        thr_dirs: dict[float, Path],
        summary: RunSummary,
    ) -> None:
        """Run inference for one file and write per-threshold, per-file outputs."""
        # Resume: skip if every threshold's CSV already exists for this file.
        if all(is_completed(d / f"{path.stem}.csv") for d in thr_dirs.values()):
            _log.info("Skipping already-processed %s", path.name)
            summary.skipped += 1
            return
        try:
            cache = runner.predict_windows(engine.run_file(path))
        except AudioError as exc:
            _log.warning("Skipping %s: %s", path.name, exc)
            summary.failed += 1
            return

        for threshold, thr_dir in thr_dirs.items():
            dets = runner.detections_at_threshold(
                cache, recording=meta, threshold=threshold, window_s=cfg.window_s, hop_s=cfg.hop_s
            )
            self._write_file_outputs(dets, thr_dir, path.stem, cfg.formats)
            summary.detections += len(dets)
        summary.processed += 1

    @staticmethod
    def _write_file_outputs(
        detections: list[Detection], thr_dir: Path, stem: str, formats: list[str]
    ) -> None:
        """Write the requested per-file output formats.

        The CSV is always written because the disk-based aggregate reads it back;
        ``parquet``/``raven`` are added when requested.
        """
        tables.write_csv(detections, thr_dir / f"{stem}.csv")
        if "parquet" in formats:
            tables.write_parquet(detections, thr_dir / f"{stem}.parquet")
        if "raven" in formats:
            raven.write_selection_table(detections, thr_dir / f"{stem}.selection.table.txt")

    def _finalize_threshold(
        self,
        threshold: float,
        thr_dir: Path,
        cfg: IdentifyConfig,
        sample_rate: int,
        config: AppConfig,
        summary: RunSummary,
    ) -> None:
        """Build the aggregate table, species summary, and optional clips."""
        csvs = sorted(p for p in thr_dir.glob("*.csv") if p.name != "all_detections.csv")
        # Drop empty (header-only) per-file CSVs before concat: at higher
        # thresholds many files have no detections, and concatenating empty
        # frames triggers a pandas dtype FutureWarning and contributes no rows.
        frames = [df for df in (pd.read_csv(p) for p in csvs if is_completed(p)) if not df.empty]
        aggregate = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=CSV_COLUMNS)
        )
        agg_path = thr_dir / "all_detections.csv"
        aggregate.to_csv(agg_path, index=False)
        summary.add_output(agg_path)

        # Human-readable per-species summary.
        if not aggregate.empty:
            counts = aggregate["label"].value_counts()
            total = len(aggregate)
            summary_lines = [f"Threshold {threshold:.2f}", f"Total detections: {total}", ""]
            summary_lines += [f"{label}\t{count}" for label, count in counts.items()]
            (thr_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

        if cfg.segments.enabled and not aggregate.empty:
            detections = _detections_from_frame(aggregate, config, threshold, cfg)
            result = extract_segments(
                detections,
                sample_rate=sample_rate,
                config=cfg.segments,
                output_dir=thr_dir / "segments",
            )
            summary.add_output(result.output_dir)


def _detections_from_frame(
    df: pd.DataFrame, config: AppConfig, threshold: float, cfg: IdentifyConfig
) -> list[Detection]:
    """Reconstruct :class:`Detection` objects from an aggregate CSV for extraction."""
    detections: list[Detection] = []
    for _, row in df.iterrows():
        meta = parse_filename(Path(row["file"]), config.filename)
        label = str(row["label"])
        detections.append(
            Detection(
                recording=meta,
                start_s=float(row["start"]),
                end_s=float(row["end"]),
                window_s=cfg.window_s,
                hop_s=cfg.hop_s,
                rank=int(row["top_k"]),
                label=label,
                common_name=label,
                species_code=label,
                confidence=float(row["confidence"]),
                threshold=threshold,
                expected_label=row.get("expected_label") or None,
            )
        )
    return detections
