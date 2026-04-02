#!/usr/bin/env python3
"""
News Video Pipeline — CLI Entry Point

Usage:
    python main.py run                      Run the full pipeline (ingest → video)
    python main.py run --tts-only           Stop after TTS, write a manifest file
    python main.py resume-video MANIFEST    Resume video from a saved manifest
    python main.py schedule                 Start the cron scheduler
    python main.py status                   Show recent run statistics
    python main.py test-ingest              Test ingestion only
    python main.py test-script              Test LLM script generation
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path so all imports resolve
sys.path.insert(0, str(Path(__file__).parent))

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(help="AI News Video Pipeline", add_completion=False)
console = Console()


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "feedparser"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def run(
    mvp: bool = typer.Option(True, help="Run in MVP mode (skip publishing/scheduling)"),
    tts_only: bool = typer.Option(False, "--tts-only", help="Stop after TTS and write a manifest for manual video work"),
    log_level: str = typer.Option("INFO", help="Logging level"),
) -> None:
    """Run the full pipeline once and generate videos.

    Use --tts-only to pause after audio generation so you can do the
    video step manually, then resume with 'resume-video'.
    """
    _setup_logging(log_level)
    from utils.hardware import log_hardware_summary

    if not mvp:
        import os
        os.environ["MVP_MODE"] = "false"

    log_hardware_summary()
    console.rule("[bold]News Video Pipeline")

    from pipeline.runner import PipelineRunner
    runner = PipelineRunner()
    ctx = runner.run(stop_after_tts=tts_only)

    if not tts_only:
        if ctx.videos:
            console.print(f"\n[bold green]Success![/bold green] {len(ctx.videos)} video(s) generated:")
            for video in ctx.videos:
                console.print(f"  → {video.output_path}")
        else:
            console.print("[yellow]No videos generated this run[/yellow]")

    if ctx.errors:
        raise typer.Exit(code=1)


@app.command(name="resume-video")
def resume_video(
    manifest: str = typer.Argument(..., help="Path to the manifest JSON from a --tts-only run"),
    log_level: str = typer.Option("INFO", help="Logging level"),
) -> None:
    """Generate videos from a saved TTS manifest.

    Run after 'python main.py run --tts-only'. You can edit the manifest
    to point audio_path entries at your own audio files before resuming.

    Example:
        python main.py run --tts-only
        # edit/replace audio files as desired
        python main.py resume-video ./output/videos/run_1_manifest.json
    """
    _setup_logging(log_level)
    console.rule("[bold]Resume Video Generation")

    from pipeline.runner import PipelineRunner
    runner = PipelineRunner()
    ctx = runner.resume_from_manifest(manifest)

    if ctx.videos:
        console.print(f"\n[bold green]Done![/bold green] {len(ctx.videos)} video(s) generated:")
        for video in ctx.videos:
            console.print(f"  → {video.output_path}")
    else:
        console.print("[yellow]No videos generated[/yellow]")

    if ctx.errors:
        raise typer.Exit(code=1)


@app.command()
def schedule(
    log_level: str = typer.Option("INFO", help="Logging level"),
) -> None:
    """Start the cron scheduler (runs continuously)."""
    _setup_logging(log_level)
    console.print("[bold]Starting scheduled pipeline (Ctrl+C to stop)[/bold]")
    from scheduling.scheduler import start_scheduler
    start_scheduler()


@app.command()
def status(
    limit: int = typer.Option(10, help="Number of recent runs to show"),
) -> None:
    """Show recent pipeline run statistics."""
    _setup_logging("WARNING")
    from storage.factory import get_storage
    from storage.models import RunRecord
    from sqlalchemy.orm import Session

    storage = get_storage()
    if not hasattr(storage, "_engine"):
        console.print("Status command only supports SQLite storage")
        return

    with Session(storage._engine) as session:
        runs = (
            session.query(RunRecord)
            .order_by(RunRecord.started_at.desc())
            .limit(limit)
            .all()
        )

    if not runs:
        console.print("No pipeline runs found in the database.")
        return

    table = Table(title=f"Last {limit} Pipeline Runs", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Started", style="dim")
    table.add_column("Status")
    table.add_column("Articles")
    table.add_column("Clusters")
    table.add_column("Videos")

    for run in runs:
        status_style = "green" if run.status == "success" else "red"
        table.add_row(
            str(run.id),
            str(run.started_at)[:16] if run.started_at else "-",
            f"[{status_style}]{run.status}[/{status_style}]",
            str(run.articles_ingested or 0),
            str(run.clusters_formed or 0),
            str(run.videos_generated or 0),
        )

    console.print(table)


@app.command(name="test-ingest")
def test_ingest(
    log_level: str = typer.Option("DEBUG", help="Logging level"),
) -> None:
    """Test ingestion stage only — no embedding, LLM, or video."""
    _setup_logging(log_level)
    import asyncio
    from ingestion.factory import build_ingestors
    from cleaning.cleaner import TextCleaner

    ingestors = build_ingestors()
    all_articles = []
    for ingestor in ingestors:
        console.print(f"Testing ingestor: [bold]{ingestor.name}[/bold]")
        articles = asyncio.run(ingestor.fetch())
        console.print(f"  → Fetched {len(articles)} raw articles")
        all_articles.extend(articles)

    if all_articles:
        cleaner = TextCleaner()
        clean = cleaner.clean(all_articles)
        console.print(f"\nCleaning result: {len(all_articles)} → {len(clean)} articles")
        if clean:
            console.print(f"\nSample article:\n  Title: {clean[0].title}")
            console.print(f"  Source: {clean[0].source}")
            console.print(f"  Words: {clean[0].word_count}")
            console.print(f"  Text preview: {clean[0].cleaned_text[:200]}...")


@app.command(name="test-script")
def test_script(
    log_level: str = typer.Option("INFO", help="Logging level"),
) -> None:
    """Test script generation with a hardcoded sample cluster."""
    _setup_logging(log_level)
    from datetime import datetime, timezone
    from cleaning.models import CleanArticle
    from clustering.models import Cluster
    from script.generator import ScriptGenerator

    # Build a sample cluster
    sample_article = CleanArticle(
        url="https://example.com/test",
        title="Scientists discover potential new treatment for Alzheimer's disease",
        source="reuters",
        timestamp=datetime.now(tz=timezone.utc),
        content_hash="test123",
        cleaned_text=(
            "Researchers at Stanford University have announced a potential breakthrough "
            "in Alzheimer's treatment. The new drug compound, tested in clinical trials "
            "on 500 patients, showed a 40% reduction in cognitive decline. The treatment "
            "works by clearing amyloid plaques from the brain. Results were published in "
            "the journal Nature Medicine."
        ),
        word_count=60,
    )

    cluster = Cluster(
        cluster_id="test-cluster",
        articles=[sample_article],
        summary=(
            "Stanford researchers published promising results for a new Alzheimer's drug "
            "that reduced cognitive decline by 40% in clinical trials."
        ),
        entities=["Stanford University", "Alzheimer's", "Nature Medicine"],
        topic="science",
    )

    console.print("[bold]Generating test script...[/bold]")
    generator = ScriptGenerator()
    script = generator.generate(cluster)
    console.print(f"\n[green]Script ({script.word_count} words, ~{script.estimated_duration_seconds:.0f}s):[/green]")
    console.print(f"\n{script.full_text}\n")


if __name__ == "__main__":
    app()
