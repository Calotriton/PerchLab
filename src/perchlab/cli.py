"""PerchLab command-line interface.

``perchlab`` with no subcommand launches interactive mode (a workflow menu +
prompts). Subcommands (``id``, ``embed``, ``benchmark``) run non-interactively
for scripting. Both paths build an :class:`AppConfig` and dispatch to the same
:class:`~perchlab.workflows.base.Workflow` objects, so behaviour never diverges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from .config import load_config
from .errors import PerchLabError
from .logging import configure_logging, get_logger
from .workflows import all_workflows, get_workflow

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Identify species, generate embeddings, and benchmark Perch V2.",
)

_log = get_logger("cli")


def _clean(overrides: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values recursively so unset CLI flags don't override config."""
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict):
            nested = _clean(value)
            if nested:
                out[key] = nested
        else:
            out[key] = value
    return out


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", help="YAML config file."),
    log_level: str | None = typer.Option(None, "--log-level", help="DEBUG/INFO/WARNING/ERROR."),
    log_file: Path | None = typer.Option(None, "--log-file", help="Write structured logs here."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Shortcut for --log-level DEBUG."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Shortcut for --log-level WARNING."),
    seed: int | None = typer.Option(None, "--seed", help="Global RNG seed."),
) -> None:
    """Configure logging/global options; launch interactive mode if no command."""
    level = log_level or ("DEBUG" if verbose else "WARNING" if quiet else "INFO")
    configure_logging(level=level, log_file=log_file)
    ctx.obj = {
        "config_path": config,
        "globals": _clean({"seed": seed, "log_level": log_level, "log_file": log_file}),
    }
    if ctx.invoked_subcommand is None:
        _run_interactive(ctx)


def _run_interactive(ctx: typer.Context) -> None:
    """Present the workflow menu and run the chosen workflow interactively."""
    from . import prompts  # local import: only needed interactively

    workflows = all_workflows()
    menu = {f"{i + 1}) {wf.name}": wf for i, wf in enumerate(workflows)}
    _log.info("Select a workflow:")
    choice = prompts.select("Workflow", choices=list(menu.keys()))
    workflow = menu[choice]

    config = load_config(ctx.obj["config_path"], ctx.obj["globals"])
    config = workflow.configure_interactive(config)
    _dispatch(workflow.command, config)


def _dispatch(command: str, config: Any) -> None:
    """Run a workflow by command, converting errors into clean exits."""
    workflow = get_workflow(command)
    try:
        workflow.run(config)
    except PerchLabError as exc:
        _log.error("%s", exc)
        raise typer.Exit(code=1) from exc


def _execute(ctx: typer.Context, command: str, overrides: dict[str, Any]) -> None:
    """Merge CLI overrides with config and run the workflow (non-interactive)."""
    merged = {**ctx.obj["globals"], **_clean(overrides)}
    config = load_config(ctx.obj["config_path"], merged)
    _dispatch(command, config)


@app.command("id")
def identify(
    ctx: typer.Context,
    input: Path = typer.Option(..., "--input", "-i", help="Input folder of recordings."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output folder."),
    window: float | None = typer.Option(None, "--window", help="Window size (s)."),
    hop: float | None = typer.Option(None, "--hop", help="Hop size (s)."),
    top_k: int | None = typer.Option(None, "--top-k", help="Predictions kept per window."),
    threshold: float | None = typer.Option(None, "--threshold", help="Confidence threshold."),
    threshold_start: float | None = typer.Option(None, "--threshold-start", help="Sweep start."),
    threshold_end: float | None = typer.Option(None, "--threshold-end", help="Sweep end."),
    threshold_step: float | None = typer.Option(None, "--threshold-step", help="Sweep step."),
    formats: str | None = typer.Option(None, "--format", help="Comma list: csv,parquet,raven."),
    extract: bool = typer.Option(False, "--extract", help="Extract audio segments."),
    extract_dir: Path | None = typer.Option(None, "--extract-dir", help="Segment output folder."),
    bin_width: float | None = typer.Option(None, "--bin-width", help="Confidence-bin width."),
    max_per_bin: int | None = typer.Option(None, "--max-per-bin", help="Max clips per bin."),
    clip_duration: float | None = typer.Option(None, "--clip-duration", help="Clip length (s)."),
    seed: int | None = typer.Option(None, "--extract-seed", help="Segment sampling seed."),
) -> None:
    """Run Species Identification (Workflow 1)."""
    sweep_on = any(v is not None for v in (threshold_start, threshold_end, threshold_step))
    overrides = {
        "identify": {
            "input_dir": input,
            "output_dir": output,
            "window_s": window,
            "hop_s": hop,
            "top_k": top_k,
            "threshold": threshold,
            "formats": formats.split(",") if formats else None,
            "sweep": {
                "enabled": True if sweep_on else None,
                "start": threshold_start,
                "end": threshold_end,
                "step": threshold_step,
            },
            "segments": {
                "enabled": True if extract else None,
                "output_dir": extract_dir,
                "bin_width": bin_width,
                "max_per_bin": max_per_bin,
                "clip_duration_s": clip_duration,
                "seed": seed,
            },
        }
    }
    _execute(ctx, "id", overrides)


@app.command("embed")
def embed(
    ctx: typer.Context,
    input: Path = typer.Option(..., "--input", "-i", help="Input folder of recordings."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output folder."),
    window: float | None = typer.Option(None, "--window", help="Window size (s)."),
    hop: float | None = typer.Option(None, "--hop", help="Hop size (s)."),
    labeled: bool = typer.Option(False, "--labeled", help="Use per-species folder labels."),
    export: str | None = typer.Option(None, "--export", help="none|parquet|npz."),
) -> None:
    """Run Embedding Generation (Workflow 2)."""
    overrides = {
        "embed": {
            "input_dir": input,
            "output_dir": output,
            "window_s": window,
            "hop_s": hop,
            "labeled": True if labeled else None,
            "export": export,
        }
    }
    _execute(ctx, "embed", overrides)


@app.command("benchmark")
def benchmark(
    ctx: typer.Context,
    input: Path = typer.Option(..., "--input", "-i", help="Labelled dataset folder."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output folder."),
    window: float | None = typer.Option(None, "--window", help="Window size (s)."),
    hop: float | None = typer.Option(None, "--hop", help="Hop size (s)."),
    threshold: float | None = typer.Option(None, "--threshold", help="Confidence threshold."),
    threshold_start: float | None = typer.Option(None, "--threshold-start", help="Sweep start."),
    threshold_end: float | None = typer.Option(None, "--threshold-end", help="Sweep end."),
    threshold_step: float | None = typer.Option(None, "--threshold-step", help="Sweep step."),
    aggregate: str | None = typer.Option(None, "--aggregate", help="window|file."),
) -> None:
    """Run Benchmark (Workflow 3)."""
    sweep_on = any(v is not None for v in (threshold_start, threshold_end, threshold_step))
    overrides = {
        "benchmark": {
            "input_dir": input,
            "output_dir": output,
            "window_s": window,
            "hop_s": hop,
            "threshold": threshold,
            "aggregate": aggregate,
            "sweep": {
                "enabled": True if sweep_on else None,
                "start": threshold_start,
                "end": threshold_end,
                "step": threshold_step,
            },
        }
    }
    _execute(ctx, "benchmark", overrides)


if __name__ == "__main__":  # pragma: no cover
    app()
